import os
import argparse
import yaml
import logging
from logging.handlers import RotatingFileHandler
import time
from tqdm import tqdm
from typing import Tuple, Dict

import pandas as pd
from pandarallel import pandarallel
import numpy as np
import matplotlib.pyplot as plt

import rdkit
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as FP
from rdkit.Chem import QED
from moses.metrics.SA_Score import sascorer
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity


def get_novelty_coverage(
    diverse_ligand: dict, 
    diverse_gen: dict, 
    div_range: list, 
    logger: logging.Logger
) -> Tuple[np.array, np.array, np.array]:
    
    """
    In a given threshold, 
    novelty is defined based on generated molecules that are structurally different from ligands,
    coverage is defined based on ligands that are structurally similar to generated molecules.
    """

    novelty = []
    coverage = []

    for threshold in tqdm(div_range):
        df_ligand = diverse_ligand[threshold]  
        df_gen = diverse_gen[threshold]        

        # Novelty part
        tanimoto_mat = np.vstack(df_gen.parallel_apply(
                lambda x: max(BulkTanimotoSimilarity(x['fp'], df_ligand['fp'].to_list())),
                axis=1,
            )
        )
        num_TP_gen = len(tanimoto_mat[tanimoto_mat <= (1-threshold)])
        current_novelty = num_TP_gen / len(df_gen)
        novelty.append(current_novelty)

        # Coverage part
        tanimoto_mat = np.vstack(df_ligand.parallel_apply(
                lambda x: max(BulkTanimotoSimilarity(x['fp'], df_gen['fp'].to_list())),
                axis=1,
            )
        )
        num_TP_ligand = len(tanimoto_mat[tanimoto_mat >= (1-threshold)])
        current_coverage = num_TP_ligand / len(df_ligand)
        coverage.append(current_coverage)

        logger.info(f"threshold {threshold} : gen {len(df_gen)}, TP gen {num_TP_gen}, ligand {len(df_ligand)}, TP ligand {num_TP_ligand}")

    novelty = np.array(novelty)
    coverage = np.array(coverage)
    f1_score = 2 * (novelty * coverage) / (novelty + coverage)

    return novelty, coverage, f1_score


def get_features(df_input: pd.DataFrame, feature_names: dict) -> pd.DataFrame:
    """
    Convert to mol object, then calculate fingerprint and feature values.
    """

    feature_list = feature_names['two_way_feature'] + feature_names['one_way_feature']

    df_input['mol'] = df_input.parallel_apply(lambda x: MolFromSmiles(x['SMILES']), axis=1)
    df_input['fp'] = df_input.parallel_apply(lambda x: FP(x['mol'], 2, nBits=2048), axis=1)

    # QED property names should be matched with rdkit specification
    # (reference : https://www.rdkit.org/docs/source/rdkit.Chem.QED.html#rdkit.Chem.QED.QEDproperties)
    df_input[['MW', 'ALOGP', 'HBA', 'HBD', 'PSA', 'ROTB', 'AROM', 'ALERTS']] = df_input.parallel_apply(
        lambda x: (QED.properties(x['mol'])), result_type="expand", axis=1
    )

    if 'SA' in feature_list:
        df_input['SA'] = df_input.parallel_apply(lambda x: sascorer.calculateScore(x['mol']), axis=1)

    columns = ['SMILES', 'mol', 'fp'] + feature_list
    return df_input[columns]


def get_thresholds(df_input: pd.DataFrame, feature_names: dict, percent: float) -> dict:
    """
    Set the criteria to determine whether or not each molecule is drug-like.
    """

    two_way_feature = feature_names['two_way_feature']
    one_way_feature = feature_names['one_way_feature']

    percentiles = [percent, (1-percent), (1-2*percent)]
    df_desc = df_input.describe(percentiles=percentiles)

    thresholds = {}
    for feature in df_input.columns[3:]:
        if feature in two_way_feature:
            minimum = str(int(percentiles[0]*100)) + "%"
            maximum = str(int(percentiles[1]*100)) + "%"
            thresholds[feature] = (df_desc[feature][minimum], df_desc[feature][maximum])

        elif feature in one_way_feature:
            maximum = str(int(percentiles[2]*100)) + "%"
            thresholds[feature] = df_desc[feature][maximum]

    return thresholds


def get_druglike_set(df_input: pd.DataFrame, feature_names: dict, thresholds: dict) -> pd.DataFrame:
    """
    Filtering out non drug-like molecules.
    """

    two_way_feature = feature_names['two_way_feature']
    one_way_feature = feature_names['one_way_feature']

    num_pass = np.zeros(len(df_input))
    for feature in df_input.columns[3:]:
        if feature in two_way_feature:
            num_pass += df_input[feature].between(thresholds[feature][0], thresholds[feature][1])
        elif feature in one_way_feature:
            num_pass += (df_input[feature] <= thresholds[feature])

    pass_idxs = np.where(np.array(num_pass) == len(df_input.columns[3:]))
    df_druglike = df_input.iloc[pass_idxs]

    return df_druglike


def get_diverse_set(df_input: pd.DataFrame, div_range: list) -> Dict[float, pd.DataFrame]:
    """
    Execute the sphere exclusion by the given threshold, then get the diverse set.
    """

    lp = rdSimDivPickers.LeaderPicker()

    picks = {}
    for threshold in tqdm(div_range):
        picks[threshold] = list(
            lp.LazyBitVectorPick(
                df_input['fp'].values, 
                len(df_input['fp'].values), 
                threshold
            )
        )
        
    diverse_set = {}
    for threshold in div_range:
        diverse_set[threshold] = df_input[['SMILES', 'fp']].iloc[picks[threshold]]
        
    return diverse_set


def aggregate_metrics(novelty: np.array, coverage: np.array, f1_score: np.array) -> Tuple[float, float, float, float]:
    """
    Based on the metric results calculated for each threshold, measure the performance of the model.
    """

    # 1. NC-AUC
    for i, (nov, cov) in enumerate(zip(novelty, coverage)):
        if i == 0:
            NC_AUC = 0
        else:
            NC_AUC += (cov - previous_cov) * (previous_nov + nov) / 2
            
        previous_nov = nov
        previous_cov = cov

    # 2. mean-precision, mean-recall
    mean_novelty = np.mean(novelty)
    mean_coverage = np.mean(coverage)

    # 3. maximum F1
    maximum_f1 = np.max(f1_score)

    return NC_AUC, mean_novelty, mean_coverage, maximum_f1


def save_agg_values(
    save_path: str, 
    NC_AUC: float, 
    mean_novelty: float, 
    mean_coverage: float, 
    maximum_f1: float
) -> None:
    
    """
    Save the performance of the model.
    """

    agg_NC = list()
    agg_NC.append(["NC_AUC", NC_AUC])
    agg_NC.append(["mean_novelty", mean_novelty])
    agg_NC.append(["mean_coverage", mean_coverage])
    agg_NC.append(["maximum_f1", maximum_f1])

    df_agg = round(pd.DataFrame(agg_NC), 3)
    df_agg.to_csv(os.path.join(save_path, "NC_agg_results.csv"), index=False, header=['metric', 'value'])

    return None

def save_div_values(
    save_path: str, 
    novelty: np.array, 
    coverage: np.array, 
    f1_score: np.array, 
    div_range: list
) -> None:
    
    """
    Save the metric results for each threshold.
    """

    df_div = pd.DataFrame()

    df_div['novelty'] = novelty
    df_div['coverage'] = coverage
    df_div['f1_score'] = f1_score

    df_div.index = div_range
    df_div.index.names = ['threshold']

    df_div = round(df_div, 3)
    df_div.to_csv(os.path.join(save_path, "NC_div_results.csv"))

    return None

def save_plot(
    save_path: str, 
    novelty: np.array, 
    coverage: np.array, 
    f1_score: np.array, 
    div_range: list
) -> None:
    
    """
    Save the figures that visualize the performance of the model.
    """

    plt.plot(div_range, novelty) 
    plt.xlabel("div")
    plt.ylabel("novelty")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(save_path, 'novelty.png'))
    plt.clf()

    plt.plot(div_range, coverage) 
    plt.xlabel("div")
    plt.ylabel("coverage")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(save_path, 'coverage.png'))
    plt.clf()

    plt.plot(div_range, f1_score) 
    plt.xlabel("div")
    plt.ylabel("f1 score")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(save_path, 'f1_score.png'))
    plt.clf()

    plt.plot(coverage, novelty) 
    plt.xlabel("coverage")
    plt.ylabel("novelty")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(save_path, 'NC_curve.png'))

    return None


if __name__ == "__main__":
    S = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='path to config file')
    parser.add_argument('--save_path', '-s', type=str, required=True, help='path to save results')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.safe_load(yml)

    os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'config.yaml'), 'w') as yml:
        yaml.dump(config, yml)
    print("save to : ", args.save_path)

    # logging
    logger = logging.getLogger(__name__)
    log_file_name = os.path.join(args.save_path, 'execution.log')
    fileHandler = RotatingFileHandler(log_file_name)
    fileHandler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] >> %(message)s'))
    logger.addHandler(fileHandler)
    logger.setLevel(logging.DEBUG)

    df_druglike = pd.read_csv(config['reference_set']['druglike_set']) 
    df_ligand = pd.read_csv(config['reference_set']['ligand_set'])     
    df_gen = pd.read_csv(config['generated_set'])
    feature_names = config['feature_names']

    # calculate features
    pandarallel.initialize(nb_workers=8, progress_bar=False, verbose=1)
    df_druglike = get_features(df_druglike, feature_names)
    df_ligand = get_features(df_ligand, feature_names)
    df_gen = get_features(df_gen, feature_names)

    percent = config['druglike_criteria']
    div_interval = config['div_interval']
    num_div = int(1/div_interval + 1)
    div_range = np.linspace(0, 1, num_div)
    div_range = np.round(div_range, 2)

    # set the drug-like criteria
    thresholds = get_thresholds(df_druglike, feature_names, percent)
    logger.info(f"thresholds = {thresholds}")

    # filtering by drug-like criteria
    df_druglike_ligand = get_druglike_set(df_ligand, feature_names, thresholds)  
    df_druglike_gen = get_druglike_set(df_gen, feature_names, thresholds)        
    logger.info(f"druglike ligand : {len(df_druglike_ligand)} ({len(df_druglike_ligand)/len(df_ligand)})")
    logger.info(f"druglike gen : {len(df_druglike_gen)} ({len(df_druglike_gen)/len(df_gen)})")

    # sphere exclusion 
    diverse_ligand = get_diverse_set(df_druglike_ligand, div_range)  
    diverse_gen = get_diverse_set(df_druglike_gen, div_range)           

    # get the novelty and coverage results
    novelty, coverage, f1_score = get_novelty_coverage(diverse_ligand, diverse_gen, div_range, logger)
    NC_AUC, mean_novelty, mean_coverage, maximum_f1 = aggregate_metrics(novelty, coverage, f1_score)

    # save the results
    save_agg_values(args.save_path, NC_AUC, mean_novelty, mean_coverage, maximum_f1)
    save_div_values(args.save_path, novelty, coverage, f1_score, div_range)
    save_plot(args.save_path, novelty, coverage, f1_score, div_range)

    E = time.time()
    logger.info(f"Total Elapsed Time : {E-S}")

