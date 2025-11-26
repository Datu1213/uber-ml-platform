"""
Setup configuration for Uber ML Platform.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="uber-ml-platform",
    version="1.0.0",
    author="Uber ML Platform Team",
    author_email="ml-platform@uber.com",
    description="Unified Enterprise ML Platform for Uber",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uber/ml-platform",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "uber-ml-train=training.train_eta_model:main",
            "uber-ml-serve=serving.main:main",
            "uber-ml-deploy=serving.deployment:deploy_model_cli",
        ],
    },
)
