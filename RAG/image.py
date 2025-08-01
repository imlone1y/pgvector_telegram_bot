import streamlit as st
import psycopg2, os, uuid, datetime, hashlib
import os
import fitz  # 用於 PDF 圖片提取
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from transformers import CLIPProcessor, CLIPModel
import hashlib
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

seen_hashes = set()
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

# 批量上傳圖片
st.header("📤 批量上傳圖片")
imgs = st.file_uploader("選擇圖片（可複選）", type=["png", "jpg", "jpeg"],
                        accept_multiple_files=True, key="img_upload")

for up in imgs:
    with st.spinner(f"處理圖片：{up.name}"):
        # ❶ 先算 hash，若重複內容直接略過
        raw      = up.getbuffer()
        img_hash = hashlib.md5(raw).hexdigest()
        cur.execute("SELECT 1 FROM documents WHERE image_hash=%s", (img_hash,))
        if cur.fetchone():
            st.info(f"⚠️ {up.name} 已存在（內容相同），跳過")
            continue

        # ❷ 生成唯一 image_ref，防止覆蓋
        unique_ref = f"{uuid.uuid4().hex}_{up.name}"
        save_path  = os.path.join(IMAGE_FOLDER, unique_ref)
        with open(save_path, "wb") as f:
            f.write(raw)

        # ❸ OCR（可失敗）
        try:
            text = pytesseract.image_to_string(Image.open(save_path),
                                               lang="chi_tra+eng").strip()
        except Exception as e:
            st.error(f"OCR 失敗：{e}")
            text = ""

        vector_str = None
        if text:
            vec        = embedding_model.embed_query(text)
            vector_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

        # ❹ 寫入資料庫（沒文字也寫，方便後續補註解）
        cur.execute("""
            INSERT INTO documents (content, embedding, source_type,
                                   image_ref, filename, image_desc,
                                   image_hash, upload_time, page_num)
            VALUES (%s,%s,'uploaded_image',%s,%s,NULL,%s,%s,NULL)
        """, (text or None, vector_str, unique_ref, up.name,
              img_hash, datetime.datetime.utcnow()))
        conn.commit()
        st.success(f"✅ 已上傳 {up.name}")


# 🔼 批量上傳文件
st.header("📤 批量上傳文件（PDF/TXT）")
docs = st.file_uploader("選擇文件（可複選）", type=["pdf", "txt"],
                        accept_multiple_files=True, key="doc_upload")

for df in docs:
    with st.spinner(f"處理文件：{df.name}"):
        doc_path = os.path.join(FILE_FOLDER, df.name)
        with open(doc_path, "wb") as f:
            f.write(df.getbuffer())

        loader = PyPDFLoader(doc_path) if df.name.endswith(".pdf") else TextLoader(doc_path)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

        # 文字分段入庫
        for doc in splitter.split_documents(raw_docs):
            content = doc.page_content.strip()
            if not content:
                continue
            vec = embedding_model.embed_query(content)
            vec_str = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
            cur.execute("""
                INSERT INTO documents (content, embedding, source_type,
                                       filename, page_num)
                VALUES (%s,%s,'pdf_text',%s,NULL)
            """, (content, vec_str, df.name))

        # --- 抽 PDF 圖片並入庫 --------------------------------------------
        if df.name.endswith(".pdf"):
            pdf_doc = fitz.open(doc_path)
            for p in range(len(pdf_doc)):
                for xref, *_ in pdf_doc[p].get_images(full=True):
                    img_bytes = pdf_doc.extract_image(xref)["image"]
                    h = hashlib.md5(img_bytes).hexdigest()
                    cur.execute("SELECT 1 FROM documents WHERE image_hash=%s", (h,))
                    if cur.fetchone():
                        continue      # 已入庫

                    img_name = f"{df.name}_p{p+1}_{xref}.png"
                    save_as  = os.path.join(IMAGE_FOLDER, img_name)
                    with open(save_as, "wb") as f:
                        f.write(img_bytes)

                    txt = pytesseract.image_to_string(Image.open(save_as),
                                                      lang="chi_tra+eng").strip()
                    vec_s = None
                    if txt:
                        vec   = embedding_model.embed_query(txt)
                        vec_s = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

                    cur.execute("""
                        INSERT INTO documents (content, embedding, source_type,
                                               filename, page_num,
                                               image_ref, image_desc,
                                               image_hash, upload_time)
                        VALUES (%s,%s,'ocr_image',%s,%s,%s,NULL,%s,%s)
                    """, (txt or None, vec_s, df.name, p+1, img_name,
                          h, datetime.datetime.utcnow()))
        conn.commit()
        st.success(f"✅ 已處理 {df.name}")


st.header("🗃️ 文件管理與刪除")

# 查詢所有出現在資料庫的檔案名稱
cur.execute("""SELECT DISTINCT filename FROM documents WHERE filename IS NOT NULL AND source_type IN ('pdf_text', 'ocr_image')
""")
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
src_type = st.radio("圖片來源", ["ocr_image", "uploaded_image"])

cur.execute("""
    SELECT DISTINCT filename, MIN(upload_time) AS t
    FROM documents
    WHERE source_type=%s
    GROUP BY filename ORDER BY t DESC
""", (src_type,))
options = [r[0] for r in cur.fetchall()]
file_sel = st.selectbox("選擇檔名", options)

if file_sel:
    cur.execute("""
        SELECT image_ref, image_desc, page_num
        FROM documents
        WHERE source_type=%s AND filename=%s
        ORDER BY page_num NULLS FIRST, upload_time
    """, (src_type, file_sel))
    rows = cur.fetchall()

    for ir, desc, pg in rows:
        img_path = os.path.join(IMAGE_FOLDER, ir)
        if not os.path.exists(img_path):
            continue
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            st.image(img_path, width=200, caption=f"頁 {pg}" if pg else "")
        with col2:
            new_desc = st.text_input("描述", value=desc or "",
                                     key=f"d_{ir}")
            if st.button("💾 儲存", key=f"s_{ir}"):
                vec_s = None
                if new_desc.strip():
                    vec   = embedding_model.embed_query(new_desc.strip())
                    vec_s = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
                cur.execute("""
                    UPDATE documents
                    SET image_desc=%s, embedding=%s
                    WHERE image_ref=%s
                """, (new_desc.strip() or None, vec_s, ir))
                conn.commit()
                st.success("已更新")

        with col3:
            if st.button("🗑 刪除", key=f"del_{ir}"):
                try:
                    # 若只剩自己一條紀錄才刪實體檔
                    cur.execute("SELECT COUNT(*) FROM documents WHERE image_ref=%s", (ir,))
                    if cur.fetchone()[0] == 1 and os.path.exists(img_path):
                        os.remove(img_path)
                    cur.execute("DELETE FROM documents WHERE image_ref=%s", (ir,))
                    conn.commit()
                    st.warning("已刪除")
                    st.rerun()
                except Exception as e:
                    st.error(f"刪除失敗：{e}")

cur.close(); conn.close()
