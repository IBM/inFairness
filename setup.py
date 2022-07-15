from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="inFairness",
    packages=['inFairness', *["inFairness." + p for p in
                              find_packages(where="./inFairness")]],
    package_dir={'':'.',},
    install_requires=[
        "tabulate~=0.8.9",
        "setuptools~=52.0.0",
        "pyyaml~=5.4.1",
        "yacs~=0.1.8",
        "torch~=1.11.0",
        "numpy~=1.22.2",
        "scikit-learn~=0.24.2",
        "cloudpickle~=2.0.0",
        "omegaconf~=2.0.6",
        "pandas~=1.3.5",
        "scipy~=1.7.3"
    ],
    description="inFairness is a Python package to train and audit individually fair PyTorch models",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version="0.1.0",
    url="https://github.com/IBM/inFairness",
    author = 'IBM Research',
    author_email = 'mayank.agarwal@ibm.com, aldo.pareja@ibm.com, onkarbhardwaj@ibm.com, mikhail.yurochkin@ibm.com',
    keywords = ['individual fairness', 'ai fairness', 'trustworthy ai', 'machine learning'],
    python_requires=">=3.8",
)
