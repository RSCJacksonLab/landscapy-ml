from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="landscapy-ml",
    version="0.1.0",
    author="Matthew A. Spence",
    author_email="matthew.spence@anu.edu.au",
    description="ML interfaces and example training pipelines for landscapy fitness landscapes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matthew-spence/landscapy-ml",
    project_urls={
        "Bug Tracker": "https://github.com/matthew-spence/landscapy-ml/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"landscapyml": ["conf/*.yaml"]},
    python_requires=">=3.10",
    install_requires=[
        # landscapy is optional here to avoid resolver conflicts when the base
        # package already exists in editable form. Use the "isolated" extra to
        # pull landscapy automatically when needed.
        "click>=8.1",
        "hydra-core>=1.3",
        "torch>=2.2",
        "pytorch-lightning>=2.3",
        "numpy>=1.24",
        "scipy>=1.10",
        "tensorboard>=2.20"
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-mock>=3.14",
            "pytest-cov>=6.0",
            "flake8>=7.0",
            "black>=24.4",
            "isort>=5.13",
            "pre-commit>=3.7",
        ],
        "graph": ["torch-geometric"],
        "gp": ["gpytorch"],
        "examples": ["torch-geometric", "gpytorch"],
        "tracking": ["wandb>=0.16"],
        "isolated": [
            # When installing stand-alone, pull landscapy from the dev branch.
            "landscapy @ git+https://github.com/RSCJacksonLab/landscapy.git@dev"
        ],
    },
)
