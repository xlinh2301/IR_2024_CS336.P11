# Backend API

## Overview

This repository contains the backend for a video search system built with FastAPI. The backend integrates with Elasticsearch for searching text data and uses backup files as a fallback mechanism. The system supports querying OCR, ASR, and object detection data. Additionally, it utilizes CLIP and FAISS for efficient image and text retrieval.

## Features

- Search text data in Elasticsearch.
- Fallback to backup files if Elasticsearch is unavailable.
- Fuzzy search support in backup data.
- Handles video metadata including video paths and frame ranges.
- Utilizes CLIP for image and text feature extraction.
- Uses FAISS for efficient similarity search and retrieval.

## Requirements

- Python 3.12 or higher
- Conda for environment management

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/xlinh2301/aic_backend.git
cd aic_backend
```
### 2. Download data and move in app/data
link: https://drive.google.com/file/d/1kJmzaSRtawGoxAGuw5lQjzHJGO6fdcFr/view?usp=drive_link

### 3. Setup with Conda Environment
Create a Conda environment with the required dependencies:

```bash
conda create --name video_search_backend python
conda activate video_search_backend
```

#### Install Dependencies
Remove faiss-cpu if you want to use faiss-gpu:
```bash
conda remove faiss-cpu
```
Then
```bash
conda install -c conda-forge -c nvidia faiss-gpu
pip install -r requirements.txt
```

### 4. Setup with Docker (Not recommended)
Build and Run Docker Containers
To build and run the Docker containers defined in docker-compose.yml, use the following command:
```bash
docker-compose up
```

### 5. Run the Application
```bash
uvicorn main:app --reload
```



