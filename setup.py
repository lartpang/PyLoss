# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="losspy",
    packages=find_packages(),
    version="0.1.0",
    license="MIT",
    description="Some loss functions for deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="lartpang",
    author_email="lartpang@gmail.com",
    url="https://github.com/lartpang/PyLoss",
    keywords=[
        "image segmentation",
        "loss",
        "deep learning",
    ],
    install_requires=[],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
