from setuptools import setup, find_packages

setup(
    name="godel-trick",
    version="0.1.0",
    packages=find_packages(include=["SAT", "VisualSudoku"]),
    install_requires=[
        "numpy>=1.25.2",
        "torch>=2.5.1",
        "typed-argument-parser>=1.10.1",
        "wandb>=0.19.0",
        "torchvision>=0.20.1",
        "timm>=1.0.12",
        "ray>=2.40.0",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.1",
    ],
)
