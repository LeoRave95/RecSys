import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RecSys",
    version="0.0.1",
    author="Davide Rossetto",
    author_email="http.davide@gmail.com",
    description="An implementation of some recommender systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davide94/RecSys",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
