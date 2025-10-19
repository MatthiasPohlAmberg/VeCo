from setuptools import setup, find_packages

setup(
    name="veco",
    version="0.1.0",
    description="Vectorization & RAG Toolkit",
    packages=find_packages(),
    python_requires=">=3.10,<3.12",
    install_requires=[
        "numpy>=1.24,<2.0",
        "scipy>=1.10,<1.12",
        "torch>=2.1,<2.3",
        "torchaudio>=2.1,<2.3",
        "torchvision>=0.16,<0.18",
        "sentence-transformers>=2.2,<3.0",
        "faiss-cpu>=1.7.4,<1.8",
        "moviepy>=1.0.3,<2.0",
        "openai-whisper>=20231117,<20250000",
        "ollama>=0.1.6,<0.3",
        "librosa>=0.10,<0.11",
        "soundfile>=0.12,<0.13",
        "webrtcvad-wheels>=2.0.10,<2.1",
        "pytesseract>=0.3.10,<0.4",
        "Pillow>=10.2,<11",
        "pdfplumber>=0.11.0,<0.12",
        "python-docx>=0.8.11,<0.9",
        "python-pptx>=0.6.23,<0.7",
        "speechbrain>=0.5.15,<0.6",
        "tqdm>=4.66,<4.67",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "ruff",
            "mypy",
        ],
        "pic-describe": [
            "transformers>=4.37,<5",
            "sentencepiece>=0.1.99,<0.2",
            "sacremoses>=0.0.53,<0.1",
        ],
    },
)
