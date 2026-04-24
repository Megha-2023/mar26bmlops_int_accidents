FROM python:3.12-slim

# Prevent python from writing pyc files 
# ENV PYTHONDONTWRITEBYTECODE = 1
# ENV PYTHONUNBUFFERED = 1

# Set working directory

WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy whole project
COPY . .

# Set python path
ENV PYTHONPATH = /app
