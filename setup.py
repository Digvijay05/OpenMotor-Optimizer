from setuptools import setup, find_packages

setup(
    name="OpenMotor-Optimizer",
    version="0.1.0",
    description="Physics-guided optimization of solid rocket motors with OpenMotor integration and GPU-accelerated search.",
    author="Simulation Engineering Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        "numba>=0.58.0",
        "torch>=2.0.0",
        "streamlit>=1.28.0",
        "pyyaml>=6.0",
        "optuna>=3.4.0"
    ],
    python_requires=">=3.10",
)
