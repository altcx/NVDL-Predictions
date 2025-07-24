"""
Setup script for NVDL Stock Predictor
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nvdl-stock-predictor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered stock prediction system using LSTM and ARIMA models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nvdl-stock-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "plotly>=5.0.0",
        "statsmodels>=0.13.0",
        "alpaca-trade-api>=2.0.0",
        "python-dotenv>=0.19.0",
        "psutil>=5.8.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "deep-learning": ["tensorflow>=2.8.0"],
        "excel": ["openpyxl>=3.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "nvdl-predictor=main:main",
        ],
    },
)