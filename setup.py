from setuptools import setup, find_packages

setup(
    name="deskvoice",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sounddevice>=0.4.6",
        "numpy>=1.24",
        "torch>=2.0",
        "torchaudio>=2.0",
        "google-genai>=1.0",
        "pydantic>=2.0",
        "python-dotenv>=1.0",
        "click>=8.0",
        "rich>=13.0",
    ],
    entry_points={
        "console_scripts": [
            "deskvoice=src.cli:main",
        ],
    },
)
