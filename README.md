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

## Project Folder Structure

Below is a brief description of the project's folder structure:
``` bash
IR
├── App
│   ├── AIC_Backend
│   ├── AIC_Frontend
├── ASR
│   ├── asr-feature-extract.ipynb
├── Extract_frame
│   ├── extract_frame.py
│   ├── extract_representative_frames.py
├── Load_to_elastic
│   ├── load-elastic-asr.ipynb
│   ├── load-elastic-ocr.ipynb
├── Scene_text
│   ├── Detection
│   ├── Recognition
├── .gitignore
├── requirements.txt
```

### Description of Key Folders and Files

- **IR**: Root folder of the project.
- **App**: Main folder containing the core components of the application.
  - **AIC_Backend**: Folder containing backend source code.
  - **AIC_Frontend**: Folder containing frontend source code.
  - **ASR**: Folder related to Automatic Speech Recognition (ASR) processing.
    - **asr-feature-extract.ipynb**: Notebook for extracting ASR features.
  - **Extract_frame**: Folder containing scripts for frame extraction.
    - **extract_frame.py**: Script to extract frames from videos.
    - **extract_representative_frames.py**: Script to extract representative frames.
  - **Load_to_elastic**: Folder containing scripts for loading data into Elasticsearch.
    - **load-elastic-asr.ipynb**: Notebook for loading ASR data into Elasticsearch.
    - **load-elastic-ocr.ipynb**: Notebook for loading OCR data into Elasticsearch.
  - **Scene_text**: Folder related to text processing in images.
    - **Detection**: Folder containing text detection source code.
    - **Recognition**: Folder containing text recognition source code.
  - **.gitignore**: File to ignore unnecessary files and folders in Git.
  - **requirements.txt**: File listing the necessary libraries to run the project.

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



