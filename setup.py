from setuptools import setup, find_packages

setup(
    name='veco',  # Name deines Moduls
    version='0.1',  # Aktuelle Version des Moduls
    packages=find_packages(where='VeCo'),
    description='A Python module for vectorizing and processing text and media files',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Matthias Pohl, Yorck Zisgen',
    author_email='kontakt@pohl-matthias.com',
    url='https://github.com/MatthiasPohlAmberg/VeCo.git',
    install_requires=[  # Liste der Abhängigkeiten
        'torch==2.8.0',
        'openai-whisper==20240930',
        'faiss-cpu==1.11.0.post1',
        'pdfplumber==0.11.7',
        'ollama==0.5.1',
        'pytesseract==0.3.13',
        'Pillow==11.0.0',
        'python-docx',
        'python-pptx==1.0.2',
        'moviepy==2.2.1',
        'sentence-transformers==5.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6, <3.12',
)
