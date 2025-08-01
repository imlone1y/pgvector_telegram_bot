# Rag + pgvector + Telegram bot 技術文檔

繁體中文 | [English](Technical_Documentation.md)

## 一、功能一覽

### 1. 網頁部分
- 批量上傳文件（PDF、TXT）  
- 批量上傳圖片  
- 查看、刪除上傳檔案  
- 單獨註解圖片與刪除上傳之圖片  

### 2. Telegram bot 部分
- 支援圖片上傳  
- 支援文件上傳  
- 以自然語言查詢資料庫內容  
- 以自然語言修改資料庫內容  

---

## 二、操作手則及注意事項

### 1. 上傳文件
將欲上傳之檔案拖曳或選取至對應區塊。  
上傳完成後，**應避免重新整理**，直到網頁右上角**無出現持續變動之圖示**，才表示上傳 + 資料向量化完成。

### 2. 上傳圖片
操作方式同上。

### 3. 刪除文件
使用下拉選單選取欲刪除文件，隨後點選刪除按鈕。  
待畫面顯示「刪除成功，請重新整理頁面。」後，方可重新整理頁面。

### 4. 圖片單獨註解
- `ocr_image` 表示從 PDF 擷取下來之圖片，下方下拉選單即為選取圖片來源文件。  
- `uploaded_image` 表示單獨上傳之圖片。  
- 所有註解內容與對應圖片皆會上傳至資料庫向量化，**每次註解完後需按下「儲存註解」按鈕**，才會儲存註解。  
- 刪除圖片會**連同其註解一併刪除**。

### 5. Telegram bot 查詢資料庫資料
使用自然語言查詢相關資料，Agent 會於資料庫中查詢相對應資料，並附上搜尋結果**最相近之圖片**。

### 6. Telegram bot 修改資料庫資料
需將欲修改資料完整敘述提供給 Agent，避免修改錯誤資料。  
修改成功後，Agent 會回覆完整修改資料，表示資料修改成功。  
> 實際修改過程如圖示。

<img width="654" height="287" alt="470206446-5d82645b-4a1a-4bb4-b1a2-5a964885ffba" src="https://github.com/user-attachments/assets/532ff734-2a83-4ed0-b5f3-f77744146a03" />

---

## 三、資料庫相關內容

### 1. 資料庫結構

#### Table: `documents`

| 欄位名稱     | 資料型別       | 說明                             |
|--------------|----------------|----------------------------------|
| id           | integer (PK)   | 主鍵，自動遞增 ID               |
| filename     | text           | 檔案名稱                         |
| page_num     | integer        | 所屬頁碼（對應 PDF 頁數）        |
| content      | text           | 擷取後的文字內容（OCR 或文本）   |
| embedding    | vector(1536)   | 向量表示（由內容或描述生成）     |
| created_at   | timestamp      | 建立時間，預設為 CURRENT_TIMESTAMP |
| image_ref    | text           | 圖片檔案名稱（實體圖片檔名）     |
| image_desc   | text           | 手動輸入的圖片描述               |
| source_type  | text           | 來源類型（如 `ocr_image`、`uploaded_image`、`pdf_text`） |
| image_hash   | varchar        | 圖片內容的 MD5 雜湊，用於檢查是否重複上傳相同圖片 |
| upload_time  | timestamp      | 實際資料寫入的時間（精確記錄每筆上傳時間，非預設欄位） |

#### Table: `upload_files`

| 欄位名稱     | 資料型別       | 說明                         |
|--------------|----------------|------------------------------|
| id           | integer (PK)   | 主鍵，自動遞增 ID           |
| file_name    | text           | 上傳檔案的檔名               |
| file_type    | text           | 檔案類型（如 `pdf`, `jpg`, `txt`） |
| upload_time  | timestamp      | 上傳時間，預設為 CURRENT_TIMESTAMP |

### 2. 資料庫內容轉移方法
#### 方法一：使用 `pg_dump` 匯出 + `psql` 匯入

```bash
# 匯出資料庫（包含表結構與資料）
pg_dump -U [使用者名稱] -h [舊主機] -p 5432 -d [資料庫名稱] > backup.sql

# 將 SQL 檔傳送至新主機
scp backup.sql user@[新主機IP]:/path/to/target

# 匯入資料到新資料庫
psql -U [使用者名稱] -h [新主機] -d [資料庫名稱] < backup.sql
```

#### 方法二：只轉移特定資料表（選擇性轉移）
```bash
# 只匯出 documents 與 upload_files 表格
pg_dump -U [使用者名稱] -h [舊主機] -t documents -t upload_files -d [資料庫名稱] > partial_backup.sql

# 匯入
psql -U [使用者名稱] -h [新主機] -d [資料庫名稱] < partial_backup.sql
```

#### 注意事項

- 若使用 pgvector，新環境需先安裝 pgvector 擴充套件：
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
- 確保新舊版本的 PostgreSQL 相容（建議同版本或以上）。
- 匯入前請建立好對應的資料庫與使用者權限。


## 四、程式碼相關

本項目分兩個檔案運行:

### 網頁 (image.py)
#### 所用技術:
- `streamlit`：網頁介面

- `pytesseract`：OCR 圖像文字辨識

- `fitz（PyMuPDF）`：提取 PDF 中的圖片

- `OpenAIEmbeddings`：產生文字向量

- `CLIPModel`：可擴充處理圖像向量

- `PostgreSQL` + `pgvector`：儲存與查詢向量資料

### Telegram bot (bot.py)
#### 所用技術:                           
- `telegram.ext` (Python Telegram Bot) 建立 Telegram bot 並處理訊息、文件、圖片等事件

- `pytesseract`                         圖片 OCR（光學文字辨識）工具，用來從圖片中提取文字

- `fitz` (PyMuPDF)                      處理 PDF 檔案、抽取每頁中的圖片 

- `OpenAIEmbeddings`                    將文字轉換為向量，用於後續比對與語意查詢

- `ChatOpenAI`                          呼叫 GPT 模型進行自然語言理解與回答
   
- `psycopg2`                            PostgreSQL 資料庫操作（寫入文字、向量、查詢結果） 

- `dotenv`                              讀取 `.env` 環境變數設定

- `json`                                將 GPT 回傳的修改格式（JSON）轉換為 Python dict 
