from setuptools import setup, find_packages

setup(
    name="alpaca-trading-algo",
    version="0.1.0",
    description="A trend-following trading algorithm using Alpaca Trading API (paper trading)",
    author="onni",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "alpaca-trade-api>=2.0.0",
        "pandas>=1.0.0",
        "numpy>=1.0.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.0.0",
        "matplotlib>=3.0.0",
        "plotly>=5.0.0",
    ],
    entry_points={
        "console_scripts": [
            "run-alpaca=main.execution:main",
        ],
    },
)
