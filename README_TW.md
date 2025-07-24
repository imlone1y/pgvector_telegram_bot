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

本項目基於 Python 程式語言，使用到外部程式庫皆在 `requirement.txt`。建議使用 [Anaconda](https://www.anaconda.com) 配置 Python 環境。需事先準備 `telegram bot token` 詳細創建 [telegram bot](https://ithelp.ithome.com.tw/m/articles/10235578) 方法。

以下設定程序已在 `macOS Seqoia v15.5` 系統上測試通過。以下為控制台/終端機（Console/Terminal/Shell）指令。

