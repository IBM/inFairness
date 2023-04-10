from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="inFairness",
    packages=[
        "inFairness",
        *["inFairness." + p for p in find_packages(where="./inFairness")],
    ],
    package_dir={"": ".",},
    install_requires=[
        "numpy>=1.21.6",
        "pandas>=1.3.5",
        "POT>=0.8.0",
        "scikit-learn>=0.24.2",
        "scipy>=1.5.4",
        "torch>=1.13.0"
    ],
    description="inFairness is a Python package to train and audit individually fair PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.2.3",
    url="https://github.com/IBM/inFairness",
    author="IBM Research",
    author_email="mayank.agarwal@ibm.com, aldo.pareja@ibm.com, onkarbhardwaj@ibm.com, mikhail.yurochkin@ibm.com",
    keywords=[
        "individual fairness",
        "ai fairness",
        "trustworthy ai",
        "machine learning",
    ],
    python_requires=">=3.7",
)
