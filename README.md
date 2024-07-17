# NC (Novelty and Coverage)
An Integrated Metric Addressing Quality Trade-offs in AI-Generated Compounds for Drug Discovery, which unifies
- Drug-likeness
- Structure Diversity
- Structure Novelty

## Data source
### MOSES
Generated sets from MOSES baseline models are equivalent to the MOSES repository: https://github.com/molecularsets/moses/tree/master/data/samples except for {model_name}--valid.csv, which only leaves SMILES that passed RDkit's validity test.

### AZ
- Paper : [SMILES-based deep generative scaffold decorator for de-novo drug design](https://chemrxiv.org/articles/SMILES-Based_Deep_Generative_Scaffold_Decorator_for_De-Novo_Drug_Design/11638383)  
- Official repository : 
https://github.com/undeadpixel/reinvent-scaffold-decorator


### MolDQN
- Paper : [Optimization of Molecules via Deep Reinforcement Learning](https://arxiv.org/abs/1810.08678)
- Official repository : 
https://github.com/google-research/google-research/tree/master/mol_dqn
- Specifically, the generated set was created using the following repository : 
https://github.com/aksub99/MolDQN-pytorch

### PGFS
- Paper : [Learning To Navigate The Synthetically Accessible Chemical Space Using Reinforcement Learning](https://arxiv.org/abs/2004.12485)
- Official repository : 
https://github.com/99andBeyond/Apollo1060
- However, the repository do not provide the model, so PGFS was implemented by referring to the contents of the paper.
