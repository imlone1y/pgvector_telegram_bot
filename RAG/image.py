import streamlit as st
import psycopg2, os, uuid, datetime, hashlib
import fitz
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

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",  # vector(3072)
    api_key=os.environ["OPENAI_API_KEY"]
)

# （可選）CLIP
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

conn = psycopg2.connect(**PG_CONF)
cur  = conn.cursor()

# =========================
# 小工具
# =========================
def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def load_image_for_preview(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path))

def rotate_pil(img: Image.Image, deg: int) -> Image.Image:
    return img if deg % 360 == 0 else img.rotate(-deg, expand=True)

def flip_pil(img: Image.Image, flip_h: bool, flip_v: bool) -> Image.Image:
    if flip_h: img = ImageOps.mirror(img)
    if flip_v: img = ImageOps.flip(img)
    return img

def save_image_overwrite(path: str, img: Image.Image) -> str:
    buf = BytesIO()
    fmt = "PNG" if path.lower().endswith(".png") else "JPEG"
    img.save(buf, format=fmt, quality=95)
    data = buf.getvalue()
    with open(path, "wb") as f: f.write(data)
    return hashlib.md5(data).hexdigest()

def vector_str_from_text(txt: str):
    if not (txt and txt.strip()):
        return None
    vec = embedding_model.embed_query(txt.strip())
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

def next_available_file_code() -> int:
    """
    取得 upload_files.file_code 的『最小缺號』，若無缺號則為 MAX+1。
    """
    cur.execute("""
        WITH mx AS (SELECT COALESCE(MAX(file_code),0) AS m FROM upload_files)
        SELECT COALESCE(
            (SELECT MIN(s) FROM generate_series(1, (SELECT m+1 FROM mx)) s
             WHERE s NOT IN (SELECT file_code FROM upload_files)),
            (SELECT m+1 FROM mx)
        )
    """)
    return int(cur.fetchone()[0])

def renumber_file_codes():
    """
    讓 upload_files.file_code 重新編號為 1..N（依 id 升冪），
    確保刪除任何一筆後，編號會自動往前遞補，沒有缺口。
    """
    cur.execute("""
        WITH seq AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn
            FROM upload_files
        )
        UPDATE upload_files u
        SET file_code = s.rn
        FROM seq s
        WHERE u.id = s.id
          AND COALESCE(u.file_code, -1) <> s.rn
    """)
    conn.commit()

def get_or_create_upload_file_by_md5(file_name: str, file_type: str, file_md5: str) -> tuple[int,int]:
    """
    以 file_md5 去重。若已存在 → 回傳 (id, file_code)；
    若不存在 → 產生最小缺號 file_code 後建立，回傳新 (id, file_code)。
    """
    cur.execute("SELECT id, file_code FROM upload_files WHERE file_md5=%s", (file_md5,))
    row = cur.fetchone()
    if row:
        return int(row[0]), int(row[1])

    code = next_available_file_code()
    cur.execute("""
        INSERT INTO upload_files (file_name, file_type, file_md5, file_code, upload_time)
        VALUES (%s,%s,%s,%s,NOW())
        RETURNING id, file_code
    """, (file_name, file_type, file_md5, code))
    rid, rcode = cur.fetchone()
    conn.commit()
    return int(rid), int(rcode)

# =========================
# UI
# =========================
st.set_page_config(page_title="向量資料管理系統", layout="wide")
st.title("📁 向量資料管理系統（以 file_code 作為對外編號）")

# -------------------------
# 批量上傳圖片（以內容 MD5 去重，file_code 最小缺號）
# -------------------------
st.header("📤 批量上傳圖片")
imgs = st.file_uploader("選擇圖片（可複選）", type=["png","jpg","jpeg"], accept_multiple_files=True, key="img_upload")

for up in imgs:
    with st.spinner(f"處理圖片：{up.name}"):
        raw       = up.getbuffer()
        file_md5  = md5_bytes(raw)
        img_md5   = file_md5
        file_type = (up.type.split("/")[-1] if up.type else "").lower() or up.name.split(".")[-1].lower()

        uf_id, code = get_or_create_upload_file_by_md5(up.name, file_type, file_md5)

        # 若此 file_code/檔案已處理過 documents → 跳過
        cur.execute("SELECT 1 FROM documents WHERE upload_file_id=%s LIMIT 1", (uf_id,))
        if cur.fetchone():
            st.info(f"⚠️ {up.name} 已存在（檔案編號 {code}），不重複寫入")
            continue

        # 存檔 + OCR + 向量
        image_ref = f"{uuid.uuid4().hex}_{up.name}"
        save_path = os.path.join(IMAGE_FOLDER, image_ref)
        with open(save_path, "wb") as f: f.write(raw)

        try:
            text = pytesseract.image_to_string(Image.open(save_path), lang="chi_tra+eng").strip()
        except Exception as e:
            st.error(f"OCR 失敗：{e}")
            text = ""

        vec_s = vector_str_from_text(text)

        cur.execute("""
            INSERT INTO documents (
                content, embedding, source_type,
                image_ref, filename, image_desc,
                image_hash, upload_time, page_num,
                upload_file_id
            )
            VALUES (%s,%s,'uploaded_image',%s,%s,NULL,%s,NOW(),NULL,%s)
        """, (text or None, vec_s, image_ref, up.name, img_md5, uf_id))
        conn.commit()
        st.success(f"✅ 已上傳 {up.name}（檔案編號 file_code: {code}）")

# -------------------------
# 批量上傳文件（PDF/TXT；file_code 最小缺號）
# -------------------------
st.header("📤 批量上傳文件（PDF/TXT）")
docs = st.file_uploader("選擇文件（可複選）", type=["pdf","txt"], accept_multiple_files=True, key="doc_upload")

for df in docs:
    with st.spinner(f"處理文件：{df.name}"):
        raw = df.getbuffer()
        with open(os.path.join(FILE_FOLDER, df.name), "wb") as f: f.write(raw)

        file_md5  = md5_bytes(raw)
        file_type = df.name.split(".")[-1].lower()
        uf_id, code = get_or_create_upload_file_by_md5(df.name, file_type, file_md5)

        cur.execute("SELECT 1 FROM documents WHERE upload_file_id=%s LIMIT 1", (uf_id,))
        if cur.fetchone():
            st.info(f"⚠️ {df.name} 已存在（檔案編號 {code}），不重複寫入")
            continue

        # 文字切塊
        loader   = PyPDFLoader(os.path.join(FILE_FOLDER, df.name)) if df.name.endswith(".pdf") else TextLoader(os.path.join(FILE_FOLDER, df.name))
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        for d in splitter.split_documents(loader.load()):
            content = d.page_content.strip()
            if not content: continue
            vec_s = vector_str_from_text(content)
            cur.execute("""
                INSERT INTO documents (
                    content, embedding, source_type,
                    filename, page_num, upload_file_id, upload_time
                )
                VALUES (%s,%s,'pdf_text',%s,NULL,%s,NOW())
            """, (content, vec_s, df.name, uf_id))

        # 若是 PDF，擷取內嵌圖片 → OCR
        if df.name.endswith(".pdf"):
            pdf_doc = fitz.open(os.path.join(FILE_FOLDER, df.name))
            for p in range(len(pdf_doc)):
                for img in pdf_doc[p].get_images(full=True):
                    xref = img[0]
                    ex   = pdf_doc.extract_image(xref)
                    img_bytes = ex["image"]
                    img_md5   = md5_bytes(img_bytes)

                    cur.execute("SELECT 1 FROM documents WHERE image_hash=%s", (img_md5,))
                    if cur.fetchone():
                        continue

                    img_name = f"{df.name}_p{p+1}_{xref}.png"
                    with open(os.path.join(IMAGE_FOLDER, img_name), "wb") as f: f.write(img_bytes)
                    try:
                        txt = pytesseract.image_to_string(Image.open(os.path.join(IMAGE_FOLDER, img_name)), lang="chi_tra+eng").strip()
                    except Exception:
                        txt = ""
                    vec_s = vector_str_from_text(txt)

                    cur.execute("""
                        INSERT INTO documents (
                            content, embedding, source_type,
                            filename, page_num, image_ref,
                            image_desc, image_hash, upload_time,
                            upload_file_id
                        )
                        VALUES (%s,%s,'ocr_image',%s,%s,%s,NULL,%s,NOW(),%s)
                    """, (txt or None, vec_s, df.name, p+1, img_name, img_md5, uf_id))

        conn.commit()
        st.success(f"✅ 已處理 {df.name}（檔案編號 file_code: {code}）")

# -------------------------
# 文件管理與刪除（以 file_code 顯示）
# -------------------------
st.header("🗃️ 文件管理與刪除")

cur.execute("""
    SELECT id, file_code, file_name, file_type,
           to_char(COALESCE(upload_time, NOW()), 'YYYY-MM-DD HH24:MI')
    FROM upload_files
    ORDER BY file_code ASC
""")
rows = cur.fetchall()
labels = [f"{r[1]} | {r[2]} ({r[3]}) @ {r[4]}" for r in rows]  # 顯示 file_code
code_to_id = {r[1]: r[0] for r in rows}

if labels:
    sel = st.selectbox("選擇要刪除的『檔案編號 file_code』", labels)
    sel_code = int(sel.split(" | ", 1)[0])
    if st.button("🗑 刪除此檔案（含其所有切塊與圖片）", key=f"delete_file_{sel_code}"):
        try:
            uf_id = code_to_id[sel_code]

            # 刪原始檔（若存在）
            cur.execute("SELECT file_name FROM upload_files WHERE id=%s", (uf_id,))
            r = cur.fetchone()
            if r:
                fp = os.path.join(FILE_FOLDER, r[0])
                if os.path.exists(fp): os.remove(fp)

            # 刪 documents 圖片檔
            cur.execute("SELECT image_ref FROM documents WHERE upload_file_id=%s AND image_ref IS NOT NULL", (uf_id,))
            for (ir,) in cur.fetchall():
                ip = os.path.join(IMAGE_FOLDER, ir)
                if os.path.exists(ip): os.remove(ip)

            # 刪記錄
            cur.execute("DELETE FROM documents WHERE upload_file_id=%s", (uf_id,))
            cur.execute("DELETE FROM upload_files WHERE id=%s", (uf_id,))
            conn.commit()
            
            renumber_file_codes()
            
            st.warning(f"❌ 已刪除『檔案編號 {sel_code}』及其所有相關資料，請重新整理")
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
