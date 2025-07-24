# pgevctor RAG telegram bot

繁體中文 | [English](README.md)

本項目為英菲克有限公司所外包之軟體系統，未經授權禁止使用、販售。

## 項目介紹

本項目連接具有 `pgvector` 額外功能之 `PostgreSQL`，與交互界面 `telegram bot`，實現強化檢索生成之功能，並使用 `OpenAIEmbeddings` 作為向量化工具，以及 `gpt-4o` 作為回模型。

上傳及管理文件部分為 `image.py`，使用 `streamlit` 部署。

## 項目結構
```tree
.
├── .env
├── bot.py
├── image_dir
├── image_index.pkl
├── image.py
├── requirements.txt
└── uploaded_files
```

## 運行指南

本項目基於 Python 程式語言，使用到外部程式庫皆在 `requirement.txt` 中。建議使用 [Anaconda](https://www.anaconda.com) 配置 Python 環境。需事先準備 `telegram bot token` 詳細創建 [telegram bot](https://ithelp.ithome.com.tw/m/articles/10235578) 方法。

以下設定程序已在 `macOS Seqoia v15.5` 系統上測試通過。以下為控制台/終端機（Console/Terminal/Shell）指令。

### 環境配置

填上 `.env` 中所需參數。

```bash
conda create -n pgvector python=3.13.3
conda activate
```

```bash
cd [該項目目錄]
pip install -r requirements.txt
```

### 運行測試

此項目需開兩個 `terminal` 同時執行，一個為網頁，另一個為 `telegram bot`。
```bash
# 文件管理網頁
cd [該項目目錄]
streamlit run image.py
```
```bash
# telegram bot server
cd [該項目目錄]
python bot.py
```

### 運行結果

- 下圖為網頁部分

<img width="1425" height="757" alt="截圖 2025-07-24 下午4 44 13" src="https://github.com/user-attachments/assets/64b34bad-fe75-4111-8860-27d58afb5683" />

- 下圖為 `telegram bot` 查詢資料對話

<img width="554" height="563" alt="截圖 2025-07-24 下午4 46 03" src="https://github.com/user-attachments/assets/a401b4d5-3377-4764-95fc-3537bbe5db9a" />

- 下圖為自然語言修改向量化資料之功能

<img width="654" height="287" alt="截圖 2025-07-24 下午4 47 15" src="https://github.com/user-attachments/assets/5d82645b-4a1a-4bb4-b1a2-5a964885ffba" />

