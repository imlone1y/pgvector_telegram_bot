# pgvector RAG Telegram Bot

[繁體中文](README_TW.md) | English

This project is a commissioned software system developed for Inphic Co., Ltd. Unauthorized use or resale is strictly prohibited.

## Project Overview

This project integrates `PostgreSQL` with the `pgvector` extension and a `Telegram bot` interface to implement Retrieval-Augmented Generation (RAG). It uses `OpenAIEmbeddings` for vectorization and `gpt-4o` as the response model. File upload and management are handled by `image.py`, which is deployed as a web interface using `Streamlit`.

The web interface includes the following features:

1. Upload and delete files
2. Upload images
3. Annotate and vectorize each image found in files (or standalone images)
4. Delete image annotations

## Project Structure

```tree
.
├── .env                 # Environment variables
├── bot.py               # Telegram bot server
├── image_dir            # Extracted images from files are stored here
├── image_index.pkl      # Pickle file for image annotation index
├── image.py             # Web interface for file management and image annotation
├── requirements.txt     # Python dependencies
└── uploaded_files       # Uploaded files are stored here
```

## Getting Started

This project is built with Python. All required libraries are listed in `requirements.txt`. It is recommended to use [Anaconda](https://www.anaconda.com) to manage the Python environment. A `Telegram bot token` is required; see [how to create a Telegram bot](https://ithelp.ithome.com.tw/m/articles/10235578) for instructions.

The following setup has been tested on `macOS Sequoia v15.5`. Use these commands in your console/terminal:

### Environment Setup

Fill in the required parameters in the `.env` file.

```bash
conda create -n pgvector python=3.13.3
conda activate pgvector
```

```bash
cd [project_directory]
pip install -r requirements.txt
```

### Running the Project

This project requires **two terminals** to run concurrently—one for the web interface and one for the Telegram bot.

```bash
# Web interface for file management
cd [project_directory]
streamlit run image.py
```

```bash
# Telegram bot server
cd [project_directory]
python bot.py
```

### Output Examples

* Screenshot of the web interface:

<img width="1425" height="757" alt="Screenshot 2025-07-24 16:44:13" src="https://github.com/user-attachments/assets/64b34bad-fe75-4111-8860-27d58afb5683" />

* Screenshot of Telegram bot query interaction:

<img width="554" height="563" alt="Screenshot 2025-07-24 16:46:03" src="https://github.com/user-attachments/assets/a401b4d5-3377-4764-95fc-3537bbe5db9a" />

* Screenshot of modifying vectorized data using natural language:

<img width="654" height="287" alt="Screenshot 2025-07-24 16:47:15" src="https://github.com/user-attachments/assets/5d82645b-4a1a-4bb4-b1a2-5a964885ffba" />
