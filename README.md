# Backend API

## Overview

This repository contains the sysytem for a Multimodal Video Retrieval. The backend integrates with Elasticsearch for searching text data and uses backup files as a fallback mechanism. The system supports querying OCR, ASR. Additionally, it utilizes CLIP and FAISS for efficient image and text retrieval.

## Features

- Search text data in Elasticsearch.
- Fallback to backup files if Elasticsearch is unavailable.
- Fuzzy search support in backup data.
- Handles video metadata including video paths and frame ranges.
- Utilizes CLIP for image and text feature extraction.
- Uses FAISS for efficient similarity search and retrieval.

## Requirements

- Python 3.12
- Conda for environment management
- Faiss GPU if needed

## Setup for backend

### 1. Clone the Repository

```bash
git clone https://github.com/xlinh2301/IR_2024_CS336.P11.git
cd App/AIC_Backend
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
```bash
pip install -r requirements.txt
conda install -c conda-forge -c nvidia faiss-gpu
```

### 4. Run the Application
```bash
uvicorn main:app --reload
```



