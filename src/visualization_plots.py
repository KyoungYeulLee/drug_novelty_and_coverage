import os
import yaml
import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gather_results(config: pd.DataFrame) -> dict:
    """ save each model result to the dictionary
    """

    models = pd.read_csv(config)
    results = {}
    for metric in ['novelty', 'coverage', 'f1_score']:
        results[metric] = OrderedDict()
    
    for model, path in zip(models['model'], models['path']):
        df = pd.read_csv(path)
        div_range = df['threshold'].values
        for metric in ['novelty', 'coverage', 'f1_score']:
            results[metric][model] = df[metric].values

    return div_range, results


def plot_results(div_range: np.array, results: dict, save_path: str) -> None:
    """ visualize novelty, coverage, f1_score, NC_curve plot
    
    """

    # single metric figs
    metric_names = ['novelty', 'coverage', 'f1_score']

    for metric in metric_names:
        for model in results[metric].keys():
            plt.plot(div_range, results[metric][model], label=model)
        plt.xlabel("div")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.legend(loc='best')
        plt.savefig(os.path.join(save_path, f'{metric}.png'))
        plt.clf()

    # AUC
    metric_1 = 'coverage'
    metric_2 = 'novelty'
    for model in results[metric_1].keys():
        plt.plot(results[metric_1][model], results[metric_2][model], label=model)
    plt.xlabel(metric_1)
    plt.ylabel(metric_2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.savefig(os.path.join(save_path, 'NC_curve.png'))

    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type=str, 
        default='plots.csv', 
        help="config csv with `model name` and `results path` columns."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help="path to save fig files"
    )
    args = parser.parse_args()

    os.makedirs(args.save_path)

    div_range, results = gather_results(args.config)
    plot_results(div_range, results, args.save_path)


