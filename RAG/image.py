import streamlit as st
import psycopg2, os, uuid, datetime, hashlib
import os
import fitz  # 用於 PDF 圖片提取
from PIL import Image, ImageOps
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from transformers import CLIPProcessor, CLIPModel
import hashlib
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from io import BytesIO

seen_hashes = set()
load_dotenv()

# =========================
# 公用工具（旋轉 / 鏡像 / 儲存 / EXIF）
# =========================
def load_image_for_preview(path: str) -> Image.Image:
    """載入圖片並依 EXIF 自動糾正方向（僅顯示/後續再旋轉）。"""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # 把 EXIF 方向轉為像素層
    return img

def rotate_pil(img: Image.Image, deg: int) -> Image.Image:
    """順時針旋轉 deg 度，expand=True 以免裁切。"""
    if deg % 360 == 0:
        return img
    return img.rotate(-deg, expand=True)  # PIL rotate 是逆時針，取負數表示順時針

def flip_pil(img: Image.Image, flip_h: bool, flip_v: bool) -> Image.Image:
    """水平/垂直鏡像翻轉。"""
    if flip_h:
        img = ImageOps.mirror(img)
    if flip_v:
        img = ImageOps.flip(img)
    return img

def save_image_overwrite(path: str, img: Image.Image) -> str:
    """覆寫存檔且移除 EXIF，回傳新檔 MD5。"""
    buf = BytesIO()
    fmt = "PNG" if path.lower().endswith(".png") else "JPEG"
    img.save(buf, format=fmt, quality=95)
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.md5(data).hexdigest()

# =========================
# 初始化資料夾與模型
# =========================
IMAGE_FOLDER = "image_dir"
FILE_FOLDER = "uploaded_files"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(FILE_FOLDER, exist_ok=True)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

# 初始化模型（如不用可移除）
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

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="向量資料管理系統", layout="wide")
st.title("📁 向量資料管理系統")

# -------------------------
# 批量上傳圖片
# -------------------------
st.header("📤 批量上傳圖片")
imgs = st.file_uploader("選擇圖片（可複選）", type=["png", "jpg", "jpeg"],
                        accept_multiple_files=True, key="img_upload")

for up in imgs:
    with st.spinner(f"處理圖片：{up.name}"):
        raw      = up.getbuffer()
        img_hash = hashlib.md5(raw).hexdigest()
        cur.execute("SELECT 1 FROM documents WHERE image_hash=%s", (img_hash,))
        if cur.fetchone():
            st.info(f"⚠️ {up.name} 已存在（內容相同），跳過")
            continue

        unique_ref = f"{uuid.uuid4().hex}_{up.name}"
        save_path  = os.path.join(IMAGE_FOLDER, unique_ref)
        with open(save_path, "wb") as f:
            f.write(raw)

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

        cur.execute("""
            INSERT INTO documents (content, embedding, source_type,
                                   image_ref, filename, image_desc,
                                   image_hash, upload_time, page_num)
            VALUES (%s,%s,'uploaded_image',%s,%s,NULL,%s,%s,NULL)
        """, (text or None, vector_str, unique_ref, up.name,
              img_hash, datetime.datetime.utcnow()))
        conn.commit()
        st.success(f"✅ 已上傳 {up.name}")

# -------------------------
# 批量上傳文件（PDF/TXT）
# -------------------------
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

        if df.name.endswith(".pdf"):
            pdf_doc = fitz.open(doc_path)
            for p in range(len(pdf_doc)):
                for xref, *_ in pdf_doc[p].get_images(full=True):
                    img_bytes = pdf_doc.extract_image(xref)["image"]
                    h = hashlib.md5(img_bytes).hexdigest()
                    cur.execute("SELECT 1 FROM documents WHERE image_hash=%s", (h,))
                    if cur.fetchone():
                        continue

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

# -------------------------
# 文件管理與刪除
# -------------------------
st.header("🗃️ 文件管理與刪除")

cur.execute("""SELECT DISTINCT filename FROM documents WHERE filename IS NOT NULL AND source_type IN ('pdf_text', 'ocr_image')
""")
doc_files = sorted([row[0] for row in cur.fetchall() if row[0]])

if doc_files:
    doc_to_delete = st.selectbox("選擇要刪除的檔案（PDF/TXT/來自 Telegram）", doc_files)
    if st.button(f"🗑 刪除文件 - {doc_to_delete}"):
        try:
            file_path = os.path.join(FILE_FOLDER, doc_to_delete)
            if os.path.exists(file_path):
                os.remove(file_path)

            cur.execute("SELECT image_ref FROM documents WHERE filename = %s AND image_ref IS NOT NULL", (doc_to_delete,))
            image_refs = [row[0] for row in cur.fetchall()]
            for image_ref in image_refs:
                img_path = os.path.join(IMAGE_FOLDER, image_ref)
                if os.path.exists(img_path):
                    os.remove(img_path)

            cur.execute("DELETE FROM documents WHERE filename = %s", (doc_to_delete,))
            conn.commit()
            st.warning(f"❌ 已刪除文件 {doc_to_delete}、相關圖片與向量紀錄，請重新整理")
        except Exception as e:
            st.error(f"刪除失敗：{e}")
else:
    st.info("目前沒有可刪除的文件。")

# -------------------------
# 圖片註解與管理（含旋轉/鏡像/覆寫）
# -------------------------
st.header("🖼 圖片註解與管理")
src_type = st.radio("圖片來源", ["ocr_image", "uploaded_image"], key="src_type")

cur.execute("""
    SELECT DISTINCT filename, MIN(upload_time) AS t
    FROM documents
    WHERE source_type=%s
    GROUP BY filename ORDER BY t DESC
""", (src_type,))
options = [r[0] for r in cur.fetchall()]
file_sel = st.selectbox("選擇檔名", options, key="file_sel")

if file_sel:
    cur.execute("""
        SELECT image_ref, image_desc, page_num
        FROM documents
        WHERE source_type=%s AND filename=%s
        ORDER BY page_num NULLS FIRST, upload_time
    """, (src_type, file_sel))
    rows = cur.fetchall()

    if "rotations" not in st.session_state:
        st.session_state["rotations"] = {}
    if "flips" not in st.session_state:
        st.session_state["flips"] = {}

    for i, (ir, desc, pg) in enumerate(rows):
        img_path = os.path.join(IMAGE_FOLDER, ir)
        if not os.path.exists(img_path):
            continue

        row_id = f"{i}_{ir}"
        key_rot = f"rot_{row_id}"
        key_flip = f"flip_{row_id}"

        if key_rot not in st.session_state["rotations"]:
            st.session_state["rotations"][key_rot] = 0
        if key_flip not in st.session_state["flips"]:
            st.session_state["flips"][key_flip] = {"h": False, "v": False}

        col1, col2, col3 = st.columns([1.2, 2.2, 1.2])

        with col1:
            base_img = load_image_for_preview(img_path)
            preview_img = rotate_pil(base_img, st.session_state["rotations"][key_rot])
            preview_img = flip_pil(preview_img,
                                   st.session_state["flips"][key_flip]["h"],
                                   st.session_state["flips"][key_flip]["v"])
            st.image(preview_img, width=220, caption=f"{ir}" + (f"（頁 {pg}）" if pg else ""))

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                if st.button("↺ 90°", key=f"rotl_{row_id}"):
                    st.session_state["rotations"][key_rot] = (st.session_state["rotations"][key_rot] + 90) % 360
                    st.rerun()
            with c2:
                if st.button("↻ 270°", key=f"rotr_{row_id}"):
                    st.session_state["rotations"][key_rot] = (st.session_state["rotations"][key_rot] + 270) % 360
                    st.rerun()
            with c3:
                if st.button("180°", key=f"rot180_{row_id}"):
                    st.session_state["rotations"][key_rot] = (st.session_state["rotations"][key_rot] + 180) % 360
                    st.rerun()
            with c4:
                if st.button("重設", key=f"rot0_{row_id}"):
                    st.session_state["rotations"][key_rot] = 0
                    st.session_state["flips"][key_flip] = {"h": False, "v": False}
                    st.rerun()
            with c5:
                if st.button("🔁 左右翻轉", key=f"fliph_{row_id}"):
                    st.session_state["flips"][key_flip]["h"] = not st.session_state["flips"][key_flip]["h"]
                    st.rerun()
            with c6:
                if st.button("🔃 上下翻轉", key=f"flipv_{row_id}"):
                    st.session_state["flips"][key_flip]["v"] = not st.session_state["flips"][key_flip]["v"]
                    st.rerun()

        with col2:
            new_desc = st.text_input("描述（可選）", value=desc or "", key=f"d_{row_id}")
            if st.button("💾 只存描述/向量", key=f"sdesc_{row_id}"):
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
                st.success("已更新描述/向量")

            if st.button("🖼 旋轉/鏡像並覆寫圖片（更新雜湊）", key=f"ssave_{row_id}"):
                try:
                    final_img = rotate_pil(load_image_for_preview(img_path), st.session_state["rotations"][key_rot])
                    final_img = flip_pil(final_img,
                                         st.session_state["flips"][key_flip]["h"],
                                         st.session_state["flips"][key_flip]["v"])
                    new_md5 = save_image_overwrite(img_path, final_img)
                    cur.execute("UPDATE documents SET image_hash=%s WHERE image_ref=%s", (new_md5, ir))

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
                    st.session_state["rotations"][key_rot] = 0
                    st.session_state["flips"][key_flip] = {"h": False, "v": False}
                    st.success("圖片已更新並覆寫，image_hash 已更新")
                    st.rerun()
                except Exception as e:
                    st.error(f"儲存失敗：{e}")

        with col3:
            if st.button("🗑 刪除", key=f"del_{row_id}"):
                try:
                    cur.execute("SELECT COUNT(*) FROM documents WHERE image_ref=%s", (ir,))
                    if cur.fetchone()[0] == 1 and os.path.exists(img_path):
                        os.remove(img_path)
                    cur.execute("DELETE FROM documents WHERE image_ref=%s", (ir,))
                    conn.commit()
                    st.warning("已刪除")
                    st.rerun()
                except Exception as e:
                    st.error(f"刪除失敗：{e}")

# -------------------------
# 關閉連線
# -------------------------
cur.close()
conn.close()
