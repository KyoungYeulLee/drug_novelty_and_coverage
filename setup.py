from setuptools import setup, find_packages

setup(
    name="novelty_coverage",
    version="v0.0.0",
    packages=find_packages(where="src/novelty_coverage"),
    python_requires='==3.7.16',
    package_dir={"": "src"},
    install_requires=[
        'pyyaml',
        'tqdm',
        'pandas',
        'pandarallel',
        'numpy',
        'matplotlib',
        'rdkit',
        'molsets'
    ]
)
