# Rag + pgvector + Telegram Bot Technical Documentation

[繁體中文](Technical_Documentation_TW.md) | English

## 1. Feature Overview

### 1.1 Web Interface
- **A.** Batch upload documents (PDF, TXT)  
- **B.** Batch upload images  
- **C.** View and delete uploaded files  
- **D.** Annotate and delete individual images  

### 1.2 Telegram Bot
- **A.** Supports image uploads  
- **B.** Supports document uploads  
- **C.** Query database contents using natural language  
- **D.** Modify database contents using natural language  

---

## 2. Instructions & Notes

### 2.1 Uploading Documents
Drag and drop or select the desired file into the upload area.  
**Do not refresh the page** until the loading spinner in the top-right corner disappears, indicating that upload and vectorization are complete.

### 2.2 Uploading Images
Same procedure as uploading documents.

### 2.3 Deleting Files
Select the file to delete using the dropdown menu, then click the "Delete" button.  
Once the message "Deleted successfully, please refresh the page." appears, you may refresh the page.

### 2.4 Image Annotation
- **A.** `ocr_image` refers to images extracted from PDF files. Use the dropdown to select the source document.  
- **B.** `uploaded_image` refers to images uploaded manually.  
- **C.** All annotations and corresponding images will be vectorized and stored in the database.  
  **Be sure to click the "Save Annotation" button** after editing, or the annotation will not be saved.  
- **D.** Deleting an image will also delete its associated annotation.

### 2.5 Querying the Database via Telegram Bot
Use natural language to query for information.  
The agent will search the database and return the most relevant data along with the closest matching image.

### 2.6 Modifying the Database via Telegram Bot
Clearly describe what you want to modify to prevent incorrect updates.  
Once modified successfully, the agent will confirm by replying with the updated data.

> Example modification process shown in illustration.

<img width="654" height="287" alt="470206446-5d82645b-4a1a-4bb4-b1a2-5a964885ffba" src="https://github.com/user-attachments/assets/20e18a1d-a73a-401a-97e9-56c8ad75f690" />

---

## 3. Database Information

### 3.1 Table Structure

#### Table: `documents`

| Column Name   | Data Type      | Description                                   |
|---------------|----------------|-----------------------------------------------|
| id            | integer (PK)   | Primary key, auto-increment ID                |
| filename      | text           | Name of the file                              |
| page_num      | integer        | Page number (corresponding to PDF page)       |
| content       | text           | Extracted text (via OCR or directly)          |
| embedding     | vector(1536)   | Embedding generated from content or description |
| created_at    | timestamp      | Creation time, default: CURRENT_TIMESTAMP     |
| image_ref     | text           | Image file name (actual file)                 |
| image_desc    | text           | Manually input image description              |
| source_type   | text           | Source type (`ocr_image`, `uploaded_image`, `pdf_text`, etc.) |
| image_hash    | varchar        | MD5 hash of image content (used to detect duplicates) |
| upload_time   | timestamp      | Actual upload timestamp when record was inserted |
| upload_file_id| integer        | seperate files                                |

indexes:
- `documents_pkey` (PK)
- `idx_documents_upload_file_id` (btree, upload_file_id)

#### Table: `upload_files`

| Column Name   | Data Type      | Description                      |
|---------------|----------------|----------------------------------|
| id            | integer (PK)   | Primary key, auto-increment ID   |
| file_name     | text           | Uploaded file name               |
| file_type     | text           | File type (`pdf`, `jpg`, `txt`)  |
| upload_time   | timestamp      | Upload time, default: CURRENT_TIMESTAMP |
| file_md5      | varchar        | context only one (MD5)           |
| file_code     | integer        | sequence id of files             |

indexes:
- `upload_files_pkey (PK)` (PK)
- `ux_upload_files_file_md5` (UNIQUE, file_md5)

---

### 3.2 How to Migrate Database Content

To migrate the database to a new environment, use PostgreSQL’s backup and restore tools as follows:

#### Method 1: Full Export & Import Using `pg_dump` + `psql`

```bash
# Export the entire database (structure + data)
pg_dump -U [username] -h [old_host] -p 5432 -d [db_name] > backup.sql

# Transfer the backup file to the new server
scp backup.sql user@[new_host_ip]:/path/to/destination

# Restore the backup to the new database
psql -U [username] -h [new_host] -d [db_name] < backup.sql
````

#### Method 2: Export Only Specific Tables

```bash
# Export only documents and upload_files tables
pg_dump -U [username] -h [old_host] -t documents -t upload_files -d [db_name] > partial_backup.sql

# Import into new server
psql -U [username] -h [new_host] -d [db_name] < partial_backup.sql
```

#### Notes

* If using `pgvector`, make sure the extension is installed in the new environment:

  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

* Make sure both PostgreSQL versions are compatible (ideally same or newer).

* Before restoring, the target database and user permissions must be properly configured.
