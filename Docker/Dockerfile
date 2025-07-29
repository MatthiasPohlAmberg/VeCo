# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# requirements.txt ins Image kopieren
COPY requirements.txt .

# Pakete installieren
RUN pip install --upgrade pip && pip install -r requirements.txt

# Projektdateien kopieren
COPY veco.py .
COPY Vectorizing_the_Company.pdf .

# Optional: Default-Startbefehl
CMD ["python", "main.py"]