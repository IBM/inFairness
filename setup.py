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
        "torch>=1.11.0",
        "numpy>=1.21.6",
        "scikit-learn>=0.24.2",
        "pandas>=1.3.5",
        "scipy>=1.5.4",
        "functorch~=0.1.1"
    ],
    description="inFairness is a Python package to train and audit individually fair PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.2.0",
    url="https://github.com/IBM/inFairness",
    author="IBM Research",
    author_email="mayank.agarwal@ibm.com, aldo.pareja@ibm.com, onkarbhardwaj@ibm.com, mikhail.yurochkin@ibm.com",
    keywords=[
        "individual fairness",
        "ai fairness",
        "trustworthy ai",
        "machine learning",
    ],
    python_requires=">=3.8",
)
