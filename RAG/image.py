import streamlit as st
import psycopg2, os, uuid, datetime, hashlib
import fitz  # PDF 圖片提取
from PIL import Image, ImageOps
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from transformers import CLIPProcessor, CLIPModel
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from io import BytesIO

# =========================
# 初始化
# =========================
load_dotenv()

IMAGE_FOLDER = "image_dir"
FILE_FOLDER  = "uploaded_files"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(FILE_FOLDER,  exist_ok=True)

PG_CONF = {
    "host": os.environ.get("PG_HOST"),
    "port": os.environ.get("PG_PORT"),
    "dbname": os.environ.get("PG_DB"),
    "user": os.environ.get("PG_USER"),
    "password": os.environ.get("PG_PASSWORD")
}

# 依你的 DB 向量維度設定；請與 documents.embedding 的 vector 維度一致
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ["OPENAI_API_KEY"]
)

# （可選）CLIP，可移除
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

conn = psycopg2.connect(**PG_CONF)
cur  = conn.cursor()

# =========================
# 公用工具（旋轉 / 鏡像 / 儲存 / EXIF）
# =========================
def load_image_for_preview(path: str) -> Image.Image:
    img = Image.open(path)
    return ImageOps.exif_transpose(img)

def rotate_pil(img: Image.Image, deg: int) -> Image.Image:
    if deg % 360 == 0:
        return img
    return img.rotate(-deg, expand=True)  # PIL 正角度=逆時針，所以取負數

def flip_pil(img: Image.Image, flip_h: bool, flip_v: bool) -> Image.Image:
    if flip_h:
        img = ImageOps.mirror(img)
    if flip_v:
        img = ImageOps.flip(img)
    return img

def save_image_overwrite(path: str, img: Image.Image) -> str:
    buf = BytesIO()
    fmt = "PNG" if path.lower().endswith(".png") else "JPEG"
    img.save(buf, format=fmt, quality=95)
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.md5(data).hexdigest()

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

# =========================
# DB 小工具（冪等：以 MD5 取得或建立 upload_file）
# =========================
def get_or_create_upload_file_by_md5(file_name: str, file_type: str, file_md5: str) -> int:
    """
    以 file_md5 做唯一鍵。若已存在則回傳原 id；否則建立新列並回傳。
    需要 DB 先建立唯一索引：
      CREATE UNIQUE INDEX IF NOT EXISTS ux_upload_files_file_md5 ON upload_files(file_md5);
    """
    cur.execute("SELECT id FROM upload_files WHERE file_md5=%s", (file_md5,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("""
        INSERT INTO upload_files (file_name, file_type, file_md5)
        VALUES (%s, %s, %s)
        ON CONFLICT (file_md5)
        DO UPDATE SET file_name = EXCLUDED.file_name
        RETURNING id
    """, (file_name, file_type, file_md5))
    fid = cur.fetchone()[0]
    conn.commit()
    return fid

def vector_str(vec) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="向量資料管理系統", layout="wide")
st.title("📁 向量資料管理系統")

# -------------------------
# 批量上傳圖片（以內容 MD5 去重）
# -------------------------
st.header("📤 批量上傳圖片")
imgs = st.file_uploader("選擇圖片（可複選）", type=["png", "jpg", "jpeg"],
                        accept_multiple_files=True, key="img_upload")

for up in imgs:
    with st.spinner(f"處理圖片：{up.name}"):
        raw      = up.getbuffer()
        file_md5 = md5_bytes(raw)                 # 用於 upload_files 去重
        img_md5  = file_md5                       # 圖片本體的雜湊

        # 取得或建立 upload_file_id（冪等）
        file_type = (up.type.split("/")[-1] if up.type else "").lower() or up.name.split(".")[-1].lower()
        upload_file_id = get_or_create_upload_file_by_md5(up.name, file_type, file_md5)

        # 若這個 ID 已處理過（documents 有資料），則直接略過重複寫入
        cur.execute("SELECT 1 FROM documents WHERE upload_file_id=%s LIMIT 1", (upload_file_id,))
        if cur.fetchone():
            st.info(f"⚠️ {up.name} 先前已處理（ID {upload_file_id}），不重複寫入")
            continue

        # 儲存圖片檔
        unique_ref = f"{uuid.uuid4().hex}_{up.name}"
        save_path  = os.path.join(IMAGE_FOLDER, unique_ref)
        with open(save_path, "wb") as f:
            f.write(raw)

        # OCR
        try:
            text = pytesseract.image_to_string(Image.open(save_path),
                                               lang="chi_tra+eng").strip()
        except Exception as e:
            st.error(f"OCR 失敗：{e}")
            text = ""

        # 向量（若有文字）
        vec_str = None
        if text:
            vec_str = "[" + ",".join(f"{v:.8f}" for v in embedding_model.embed_query(text)) + "]"

        # 寫入 documents
        cur.execute("""
            INSERT INTO documents (
                content, embedding, source_type,
                image_ref, filename, image_desc,
                image_hash, upload_time, page_num,
                upload_file_id
            )
            VALUES (%s,%s,'uploaded_image',%s,%s,NULL,%s,%s,NULL,%s)
        """, (text or None, vec_str, unique_ref, up.name,
              img_md5, datetime.datetime.utcnow(), upload_file_id))
        conn.commit()
        st.success(f"✅ 已上傳 {up.name}（檔案 ID: {upload_file_id}）")

# -------------------------
# 批量上傳文件（PDF/TXT，以內容 MD5 去重）
# -------------------------
st.header("📤 批量上傳文件（PDF/TXT）")
docs = st.file_uploader("選擇文件（可複選）", type=["pdf", "txt"],
                        accept_multiple_files=True, key="doc_upload")

for df in docs:
    with st.spinner(f"處理文件：{df.name}"):
        # 先把整份檔案寫到暫存路徑取得 bytes 以計算 MD5
        doc_path = os.path.join(FILE_FOLDER, df.name)
        raw = df.getbuffer()
        with open(doc_path, "wb") as f:
            f.write(raw)
        file_md5 = md5_bytes(raw)  # 用於 upload_files 冪等

        file_type = df.name.split(".")[-1].lower()
        upload_file_id = get_or_create_upload_file_by_md5(df.name, file_type, file_md5)

        # 若此 ID 已有 documents，視為處理過，直接略過切塊與寫入
        cur.execute("SELECT 1 FROM documents WHERE upload_file_id=%s LIMIT 1", (upload_file_id,))
        if cur.fetchone():
            st.info(f"⚠️ {df.name} 先前已處理（ID {upload_file_id}），不重複寫入")
            continue

        # 文字切塊
        loader   = PyPDFLoader(doc_path) if df.name.endswith(".pdf") else TextLoader(doc_path)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

        for doc in splitter.split_documents(raw_docs):
            content = doc.page_content.strip()
            if not content:
                continue
            vec_str = "[" + ",".join(f"{x:.8f}" for x in embedding_model.embed_query(content)) + "]"
            cur.execute("""
                INSERT INTO documents (
                    content, embedding, source_type,
                    filename, page_num, upload_file_id, upload_time
                )
                VALUES (%s,%s,'pdf_text',%s,NULL,%s,%s)
            """, (content, vec_str, df.name, upload_file_id, datetime.datetime.utcnow()))

        # 若是 PDF，擷取內嵌圖片 → OCR → 寫入 documents（同一 upload_file_id）
        if df.name.endswith(".pdf"):
            pdf_doc = fitz.open(doc_path)
            for p in range(len(pdf_doc)):
                for img in pdf_doc[p].get_images(full=True):
                    xref = img[0]
                    extracted = pdf_doc.extract_image(xref)
                    img_bytes = extracted["image"]
                    img_md5   = md5_bytes(img_bytes)

                    # 圖片內容去重
                    cur.execute("SELECT 1 FROM documents WHERE image_hash=%s", (img_md5,))
                    if cur.fetchone():
                        continue

                    img_name = f"{df.name}_p{p+1}_{xref}.png"
                    save_as  = os.path.join(IMAGE_FOLDER, img_name)
                    with open(save_as, "wb") as f:
                        f.write(img_bytes)

                    try:
                        txt = pytesseract.image_to_string(Image.open(save_as),
                                                          lang="chi_tra+eng").strip()
                    except Exception as e:
                        st.error(f"OCR 失敗：{e}")
                        txt = ""

                    vec_s = None
                    if txt:
                        vec_s = "[" + ",".join(f"{v:.8f}" for v in embedding_model.embed_query(txt)) + "]"

                    cur.execute("""
                        INSERT INTO documents (
                            content, embedding, source_type,
                            filename, page_num,
                            image_ref, image_desc,
                            image_hash, upload_time,
                            upload_file_id
                        )
                        VALUES (%s,%s,'ocr_image',%s,%s,%s,NULL,%s,%s,%s)
                    """, (txt or None, vec_s, df.name, p+1, img_name,
                          img_md5, datetime.datetime.utcnow(), upload_file_id))
        conn.commit()
        st.success(f"✅ 已處理 {df.name}（檔案 ID: {upload_file_id}）")

# -------------------------
# 文件管理與刪除（以 upload_files 為主）
# -------------------------
st.header("🗃️ 文件管理與刪除")

cur.execute("""
    SELECT id, file_name, file_type, to_char(COALESCE(upload_time, NOW()), 'YYYY-MM-DD HH24:MI')
    FROM upload_files
    ORDER BY id DESC
""")
uf_rows   = cur.fetchall()
uf_labels = [f"{r[0]} | {r[1]} ({r[2]}) @ {r[3]}" for r in uf_rows]
uf_id_map = {lab: rid for lab, (rid, *_rest) in zip(uf_labels, uf_rows)}

if uf_labels:
    sel = st.selectbox("選擇要刪除的『檔案 ID』", uf_labels)
    if st.button("🗑 刪除此檔案（含其所有切塊與圖片）"):
        try:
            sel_id = uf_id_map[sel]

            # 刪除原始檔（若存在）
            cur.execute("SELECT file_name FROM upload_files WHERE id=%s", (sel_id,))
            r = cur.fetchone()
            if r:
                fp = os.path.join(FILE_FOLDER, r[0])
                if os.path.exists(fp):
                    os.remove(fp)

            # 刪掉 documents 圖片檔
            cur.execute("SELECT image_ref FROM documents WHERE upload_file_id=%s AND image_ref IS NOT NULL", (sel_id,))
            for (ir,) in cur.fetchall():
                ip = os.path.join(IMAGE_FOLDER, ir)
                if os.path.exists(ip):
                    os.remove(ip)

            # 刪記錄
            cur.execute("DELETE FROM documents   WHERE upload_file_id=%s", (sel_id,))
            cur.execute("DELETE FROM upload_files WHERE id=%s", (sel_id,))
            conn.commit()
            st.warning(f"❌ 已刪除『檔案 ID {sel_id}』及其所有相關資料，請重新整理")
        except Exception as e:
            st.error(f"刪除失敗：{e}")
else:
    st.info("目前沒有可刪除的檔案。")

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
        if not ir:
            continue
        img_path = os.path.join(IMAGE_FOLDER, ir)
        if not os.path.exists(img_path):
            continue

        row_id  = f"{i}_{ir}"
        key_rot = f"rot_{row_id}"
        key_flip= f"flip_{row_id}"

        st.session_state["rotations"].setdefault(key_rot, 0)
        st.session_state["flips"].setdefault(key_flip, {"h": False, "v": False})

        col1, col2, col3 = st.columns([1.2, 2.2, 1.2])

        with col1:
            base_img   = load_image_for_preview(img_path)
            preview_img= rotate_pil(base_img, st.session_state["rotations"][key_rot])
            preview_img= flip_pil(preview_img,
                                  st.session_state["flips"][key_flip]["h"],
                                  st.session_state["flips"][key_flip]["v"])
            st.image(preview_img, width=220, caption=f"{ir}" + (f"（頁 {pg}）" if pg else ""))

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                if st.button("↺ 90°",   key=f"rotl_{row_id}"):
                    st.session_state["rotations"][key_rot] = (st.session_state["rotations"][key_rot] + 90) % 360
                    st.rerun()
            with c2:
                if st.button("↻ 270°",  key=f"rotr_{row_id}"):
                    st.session_state["rotations"][key_rot] = (st.session_state["rotations"][key_rot] + 270) % 360
                    st.rerun()
            with c3:
                if st.button("180°",    key=f"rot180_{row_id}"):
                    st.session_state["rotations"][key_rot] = (st.session_state["rotations"][key_rot] + 180) % 360
                    st.rerun()
            with c4:
                if st.button("重設",     key=f"rot0_{row_id}"):
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
                    vec_s = "[" + ",".join(f"{v:.8f}" for v in embedding_model.embed_query(new_desc.strip())) + "]"
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
                        vec_s = "[" + ",".join(f"{v:.8f}" for v in embedding_model.embed_query(new_desc.strip())) + "]"
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
