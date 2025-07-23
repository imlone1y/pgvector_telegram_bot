import streamlit as st
import psycopg2
import os
import fitz  # 用於 PDF 圖片提取
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from transformers import CLIPProcessor, CLIPModel

load_dotenv()

# 初始化資料夾與模型
IMAGE_FOLDER = "image_dir"
FILE_FOLDER = "uploaded_files"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(FILE_FOLDER, exist_ok=True)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

# 初始化模型
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

PG_CONF = {
    "host": os.environ.get("PG_HOST"),
    "port": os.environ.get("PG_PORT"),
    "dbname": os.environ.get("PG_DB"),
    "user": os.environ.get("PG_USER"),
    "password": os.environ.get("PG_PASSWORD")
}

conn = psycopg2.connect(**PG_CONF)
cur = conn.cursor()



st.set_page_config(page_title="向量資料管理系統", layout="wide")
st.title("📁 向量資料管理系統")

# 🔼 批量上傳圖片
st.header("📤 批量上傳圖片")
uploaded_images = st.file_uploader("選擇圖片（可複選）", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="img_upload")

for uploaded_file in uploaded_images:
    save_path = os.path.join(IMAGE_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"已儲存圖片：{uploaded_file.name}")

    cur.execute("SELECT 1 FROM documents WHERE image_ref = %s", (uploaded_file.name,))
    if not cur.fetchone():
        image = Image.open(save_path)
        try:
            import pytesseract
            text = pytesseract.image_to_string(image, lang="chi_tra+eng").strip()
            if text:
                vector = embedding_model.embed_query(text)
                vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
                cur.execute(
                    """
                    INSERT INTO documents (content, embedding, source_type, image_ref, image_desc, filename, page_num)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (text, vector_str, 'uploaded_image', uploaded_file.name, None, uploaded_file.name, None)
                )
        except Exception as e:
            st.error(f"OCR/嵌入失敗：{e}")
conn.commit()

# 🔼 批量上傳文件
st.header("📤 批量上傳文件（PDF/TXT）")
uploaded_docs = st.file_uploader("選擇文件（可複選）", type=["pdf", "txt"], accept_multiple_files=True, key="doc_upload")

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

for doc_file in uploaded_docs:
    doc_path = os.path.join(FILE_FOLDER, doc_file.name)
    with open(doc_path, "wb") as f:
        f.write(doc_file.getbuffer())
    st.success(f"✅ 已儲存文件：{doc_file.name}")

    if doc_file.name.endswith(".pdf"):
        loader = PyPDFLoader(doc_path)
    else:
        loader = TextLoader(doc_path)

    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    documents = splitter.split_documents(raw_docs)

    for doc in documents:
        content = doc.page_content.strip()
        if not content:
            continue
        vector = embedding_model.embed_query(content)
        vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
        cur.execute(
            """
            INSERT INTO documents (content, embedding, source_type, filename, page_num)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (content, vector_str, 'pdf_text', doc_file.name, None)
        )

    # 補充：處理 PDF 圖片 OCR（比照 bot.py）
    if doc_file.name.endswith(".pdf"):
        pdf_doc = fitz.open(doc_path)
        for page_index in range(len(pdf_doc)):
            images = pdf_doc[page_index].get_images(full=True)
            for img_index, img_info in enumerate(images):
                xref = img_info[0]
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]

                image_filename = f"{doc_file.name}_page{page_index+1}_img{img_index+1}.png"
                image_path = os.path.join(IMAGE_FOLDER, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                image = Image.open(image_path)
                import pytesseract
                text = pytesseract.image_to_string(image, lang="chi_tra+eng").strip()

                if text:
                    vector = embedding_model.embed_query(text)
                    vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
                    cur.execute(
                        """
                        INSERT INTO documents (content, embedding, source_type, filename, page_num, image_ref, image_desc)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (text, vector_str, 'ocr_image', doc_file.name, page_index+1, image_filename, None)
                    )
conn.commit()

st.header("🗃️ 文件管理與刪除")

# 查詢所有出現在資料庫的檔案名稱
cur.execute("SELECT DISTINCT filename FROM documents WHERE filename IS NOT NULL")
doc_files = sorted([row[0] for row in cur.fetchall() if row[0]])

if doc_files:
    doc_to_delete = st.selectbox("選擇要刪除的檔案（PDF/TXT/來自 Telegram）", doc_files)
    if st.button(f"🗑 刪除文件 - {doc_to_delete}"):
        try:
            # 刪除實體 PDF/TXT 檔案（網站上傳路徑）
            file_path = os.path.join(FILE_FOLDER, doc_to_delete)
            if os.path.exists(file_path):
                os.remove(file_path)

            # 查詢 OCR 圖片（image_ref）並刪除實體圖檔
            cur.execute("SELECT image_ref FROM documents WHERE filename = %s AND image_ref IS NOT NULL", (doc_to_delete,))
            image_refs = [row[0] for row in cur.fetchall()]
            for image_ref in image_refs:
                img_path = os.path.join(IMAGE_FOLDER, image_ref)
                if os.path.exists(img_path):
                    os.remove(img_path)

            # 刪除所有與該 filename 相關的紀錄
            cur.execute("DELETE FROM documents WHERE filename = %s", (doc_to_delete,))
            conn.commit()
            st.warning(f"❌ 已刪除文件 {doc_to_delete}、相關圖片與向量紀錄，請重新整理")
        except Exception as e:
            st.error(f"刪除失敗：{e}")
else:
    st.info("目前沒有可刪除的文件。")

# 📋 圖片註解功能（依來源與檔案過濾）
st.header("🖼 圖片註解與管理")
source_type = st.radio("選擇圖片來源類型", ["ocr_image", "uploaded_image"])

cur.execute("SELECT DISTINCT filename FROM documents WHERE source_type = %s", (source_type,))
available_files = [row[0] for row in cur.fetchall() if row[0]]
selected_file = st.selectbox("選擇來源檔案（PDF或圖片檔）", available_files)

if selected_file:
    if source_type == "ocr_image":
        cur.execute("""
            SELECT image_ref, image_desc, page_num FROM documents
            WHERE source_type = %s AND filename = %s
            ORDER BY page_num ASC, image_ref ASC
        """, (source_type, selected_file))
    else:
        cur.execute("""
            SELECT image_ref, image_desc, page_num FROM documents
            WHERE source_type = %s AND filename = %s
            ORDER BY image_ref ASC
        """, (source_type, selected_file))

    rows = cur.fetchall()

    for image_ref, image_desc, page_num in rows:
        image_path = os.path.join(IMAGE_FOLDER, image_ref)
        if not os.path.exists(image_path):
            continue
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.image(image_path, width=200, caption=f"頁碼 {page_num}" if page_num else None)
        with col2:
            st.markdown(f"**檔名：** `{image_ref}`")
            new_desc = st.text_input(f"輸入描述", value=image_desc or "", key=f"desc_{image_ref}")
            if st.button(f"💾 儲存註解 - {image_ref}", key=f"save_{image_ref}"):
                clean_desc = new_desc.strip()
                cur.execute("UPDATE documents SET image_desc = %s WHERE image_ref = %s", 
                            (clean_desc if clean_desc else None, image_ref))

                if clean_desc and clean_desc != (image_desc or ""):
                    # 有新註解 → 生成向量
                    vector = embedding_model.embed_query(clean_desc)
                    vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
                    cur.execute("UPDATE documents SET embedding = %s WHERE image_ref = %s", (vector_str, image_ref))
                elif not clean_desc:
                    # 沒有註解 → 把向量設為 NULL（防止被查到）
                    cur.execute("UPDATE documents SET embedding = NULL WHERE image_ref = %s", (image_ref,))
                
                conn.commit()
                st.success(f"✅ 已更新 {image_ref} 的註解與向量")
        with col3:
            if st.button(f"🗑 刪除圖片 - {image_ref}", key=f"delete_{image_ref}"):
                try:
                    os.remove(image_path)
                    cur.execute("DELETE FROM documents WHERE image_ref = %s", (image_ref,))
                    conn.commit()
                    st.warning(f"❌ 已刪除圖片 {image_ref} 與其向量紀錄，請重新整理")
                except Exception as e:
                    st.error(f"刪除失敗：{e}")
    
cur.close()
conn.close()
