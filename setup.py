from setuptools import setup

with open("README.md","r",encoding ="utf-8") as f:
    long_description = f.read()

setup(
    name ="src",
    version="0.0.1",
    author="Rahul_P",
    description="Small Package for ANN-implementation",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/rahulpaygude02/ANN-implementation.git",
    author_email ="rahulpaygude8293@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires = [
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
        "PyYAML"
        ]

)