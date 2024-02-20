from setuptools import find_packages, setup

setup(
    name="pair_encoder",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "fire",
        "scipy",
        "torch",
        "tqdm",
        "transformers",
        "pandas",
        "numpy",
        "scikit-learn",
    ],
    author="Tollef Jørgensen",
    author_email="tollefj@gmail.com",
    description="STS augmented pair encoder",
    url="https://github.com/tollefj/STS-augmented-pair-encoder",
)
