from setuptools import setup, find_packages

setup(
    name="deskvoice",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "sounddevice>=0.4.6",
        "numpy>=1.24",
        "torch>=2.0",
        "torchaudio>=2.0",
        "google-genai>=1.0",
        "openai>=1.0",
        "pydantic>=2.0",
        "python-dotenv>=1.0",
        "click>=8.0",
        "rich>=13.0",
        "packaging>=21.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.27.0",
        "textual>=0.50.0",
    ],
    extras_require={
        "speaker": ["resemblyzer>=0.1.3"],
    },
    entry_points={
        "console_scripts": [
            "deskvoice=src.cli:main",
        ],
    },
)
