from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="umm-quantum",
    version="0.1.0",
    author="Justin Hart",
    author_email="justin@viridis.llc",
    description="Universal Measurement Machine: AI-programmed adaptive quantum measurement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/viridis-llc/umm-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "qutip>=4.6.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "jsonschema>=3.2.0",
        "tqdm>=4.62.0",
        "pytest>=6.2.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pytest-cov>=2.12.0",
            "sphinx>=4.0.0",
        ],
    },
)
