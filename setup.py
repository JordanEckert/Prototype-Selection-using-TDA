"""
Setup script for TPS Bifiltration Prototype Selector

Install with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tps-bifiltration",
    version="1.0.0",
    author="Jordan Eckert, Elvan Ceyhan, Henry Schenck",
    author_email="jpe0018@auburn.edu",
    description="Topological Prototype Selection using Bifiltration and Persistent Homology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tps-bifiltration",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.3.0",
        "ripser>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    keywords="topological data analysis, persistent homology, prototype selection, "
             "machine learning, data reduction, TDA, bifiltration",
    project_urls={
        "Paper": "https://arxiv.org/abs/xxxx.xxxxx",  # Update with actual arXiv link when available
        "Bug Reports": "https://github.com/yourusername/tps-bifiltration/issues",
        "Source": "https://github.com/yourusername/tps-bifiltration",
    },
)
