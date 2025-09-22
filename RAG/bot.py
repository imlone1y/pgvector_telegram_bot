import os, re, uuid, hashlib, datetime, tempfile, json
from io import BytesIO
import psycopg2, fitz, pytesseract
from PIL import Image
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ---------- 基礎設定 --------------------------------------------------------
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
IMAGE_DIR = "image_dir";       os.makedirs(IMAGE_DIR, exist_ok=True)
FILE_DIR  = "uploaded_files";  os.makedirs(FILE_DIR, exist_ok=True)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",          # ← 請確認和 DB 的向量維度一致
    api_key=os.getenv("OPENAI_API_KEY")
)
llm = ChatOpenAI(model="gpt-5", temperature=1)

STRICT_SYS_PROMPT = """1. 僅使用下方 Context 的內容回答；不得加入外部常識或臆測。
2. 若 Context 無法支持答案，請直接回覆：「我不知道。需要更多資訊。」不要在答案內加入任何引用或標註。
3. 禁止虛構數字、名詞定義與結論；對專有名詞保留原文。
4. 答案力求精確、簡潔。"""

PG_CONF = dict(
    host=os.environ["PG_HOST"],
    port=os.environ["PG_PORT"],
    dbname=os.environ["PG_DB"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"]
)

# 回答時使用幾段、以及距離門檻（<=> 越小越相似）
SRC_TOPK = int(os.getenv("SRC_TOPK", "2"))
SRC_SIM_THRESHOLD  = float(os.getenv("SRC_SIM_THRESHOLD", "0.60"))
SRC_FALLBACK_MAX   = float(os.getenv("SRC_FALLBACK_MAX",  "0.85"))

# 僅允許「ID: 問題」格式
ID_QUERY_RE = re.compile(r"^\s*(\d+)\s*[:：]\s*(.+)$")

# ---------- 共用小函式 ------------------------------------------------------
def db_conn():
    return psycopg2.connect(**PG_CONF)

def vector_to_str(vec):
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def list_files(limit: int = 100):
    sql = """
    SELECT id, file_name, file_type,
           COALESCE(to_char(upload_time,'YYYY-MM-DD HH24:MI'), '')
    FROM upload_files
    ORDER BY id DESC
    LIMIT %s
    """
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (limit,))
        return cur.fetchall()

def list_files_text(limit: int = 100) -> str:
    rows = list_files(limit)
    if not rows:
        return "目前沒有任何檔案。請先上傳。"
    lines = ["可查詢的檔案清單（輸入 `ID: 問題` 開始查詢）："]
    for rid, name, ftype, ts in rows:
        lines.append(f"• {rid} | {name} ({ftype}) {ts}")
    return "\n".join(lines)

def get_or_create_upload_file_by_md5(file_name: str, file_type: str, file_md5: str) -> int:
    """
    以 file_md5 做唯一鍵。需要 DB 先有：
      ALTER TABLE upload_files ADD COLUMN IF NOT EXISTS file_md5 varchar;
      CREATE UNIQUE INDEX IF NOT EXISTS ux_upload_files_file_md5 ON upload_files(file_md5);
    """
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM upload_files WHERE file_md5=%s", (file_md5,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute("""
            INSERT INTO upload_files (file_name, file_type, file_md5, upload_time)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (file_md5) DO UPDATE SET file_name = EXCLUDED.file_name
            RETURNING id
        """, (file_name, file_type, file_md5))
        fid = cur.fetchone()[0]
        conn.commit()
        return fid

def insert_pdf_text(cur, content, filename, upload_file_id: int):
    vec_str = vector_to_str(embedding_model.embed_query(content))
    cur.execute("""
        INSERT INTO documents (content, embedding, source_type,
                               filename, upload_time, upload_file_id)
        VALUES (%s,%s,'pdf_text',%s,%s,%s)
    """, (content, vec_str, filename, datetime.datetime.utcnow(), upload_file_id))

def save_image_and_insert(cur, img_bytes, ocr_text,
                          source_type, filename, upload_file_id: int, page_num=None):
    # 依圖片內容去重
    img_hash = md5_bytes(img_bytes)
    cur.execute("SELECT 1 FROM documents WHERE image_hash=%s", (img_hash,))
    if cur.fetchone():
        return

    image_ref = f"{uuid.uuid4().hex}_{filename}"
    img_path  = os.path.join(IMAGE_DIR, image_ref)
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    vec_str = None
    if ocr_text:
        vec_str = vector_to_str(embedding_model.embed_query(ocr_text))

    cur.execute("""
        INSERT INTO documents (content, embedding, source_type,
                               filename, page_num,
                               image_ref, image_hash, upload_time,
                               upload_file_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (ocr_text or None, vec_str, source_type,
          filename, page_num,
          image_ref, img_hash, datetime.datetime.utcnow(),
          upload_file_id))

def search_by_file_id(file_id: int, query: str, k: int = 8):
    """僅在指定 upload_file_id 範圍內檢索內容段落。"""
    qvec = embedding_model.embed_query(query)
    qstr = vector_to_str(qvec)
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT content, filename, page_num, source_type,
                   (embedding <=> %s) AS dist
            FROM documents
            WHERE upload_file_id = %s
              AND embedding IS NOT NULL
            ORDER BY dist ASC
            LIMIT %s
        """, (qstr, file_id, k))
        return cur.fetchall()

def build_rag_prompt(query: str, rows):
    # rows: (content, filename, page_num, source_type, dist)
    context_parts, sources = [], []
    for content, filename, page_num, source_type, dist in rows:
        if not content or dist is None or dist > SRC_SIM_THRESHOLD:
            continue
        context_parts.append(content)
        src = f"{filename}"
        if page_num: src += f" p.{page_num}"
        if source_type: src += f" ({source_type})"
        sources.append(src)
        if len(context_parts) >= SRC_TOPK:
            break

    if not context_parts and rows:
        # 保底：拿最相似的一段，但距離要 <= SRC_FALLBACK_MAX
        best_content, best_filename, best_page, best_type, best_dist = rows[0]
        if best_content and best_dist is not None and best_dist <= SRC_FALLBACK_MAX:
            context_parts.append(best_content)
            src = f"{best_filename}"
            if best_page: src += f" p.{best_page}"
            if best_type: src += f" ({best_type})"
            sources.append(src)

    context = "\n".join(context_parts)
    messages = [
        {"role": "system", "content": STRICT_SYS_PROMPT},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user",   "content": query}
    ]
    return context, messages, sources

# ---------- PDF 上傳處理（含 MD5 冪等） ------------------------------------
async def process_pdf(file_path):
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_md5 = md5_bytes(pdf_bytes)

    upload_file_id = get_or_create_upload_file_by_md5(filename, "pdf", pdf_md5)

    # 若此 ID 之前已處理過，就直接略過
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE upload_file_id=%s LIMIT 1", (upload_file_id,))
        if cur.fetchone():
            return upload_file_id

    loader   = PyPDFLoader(file_path)
    pdf_doc  = fitz.open(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    with db_conn() as conn:
        with conn.cursor() as cur:
            # 文字分段
            for doc in splitter.split_documents(loader.load()):
                content = (doc.page_content or "").strip()
                if content:
                    insert_pdf_text(cur, content, filename, upload_file_id)

            # 圖片 OCR
            for p in range(len(pdf_doc)):
                for xref, *_ in pdf_doc[p].get_images(full=True):
                    img_bytes = pdf_doc.extract_image(xref)["image"]
                    ocr_text  = pytesseract.image_to_string(
                        Image.open(BytesIO(img_bytes)), lang="chi_tra+eng"
                    ).strip()
                    save_image_and_insert(cur, img_bytes, ocr_text,
                                          'ocr_image', filename, upload_file_id, page_num=p+1)
        conn.commit()
    return upload_file_id

# ---------- 圖片上傳處理（含 MD5 冪等） ------------------------------------
async def process_photo(img_bytes, original_name):
    img_md5 = md5_bytes(img_bytes)
    upload_file_id = get_or_create_upload_file_by_md5(original_name, "image", img_md5)

    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE upload_file_id=%s LIMIT 1", (upload_file_id,))
        if cur.fetchone():
            return upload_file_id

    ocr_text = pytesseract.image_to_string(
        Image.open(BytesIO(img_bytes)), lang="chi_tra+eng"
    ).strip()

    with db_conn() as conn:
        with conn.cursor() as cur:
            save_image_and_insert(cur, img_bytes, ocr_text,
                                  'uploaded_image', original_name, upload_file_id)
        conn.commit()
    return upload_file_id

# ---------- 文字問答（強制 ID: 查詢） --------------------------------------
async def qa_with_file_scope(fid: int, user_query: str, send_text):
    await send_text("請稍等，正在查詢資料中...")

    rows = search_by_file_id(fid, user_query, k=8)
    context, messages, sources = build_rag_prompt(user_query, rows)

    if not context.strip():
        await send_text("我不知道。需要更多資訊。")
        await send_text(list_files_text())  # 回清單形成循環
        return

    try:
        resp = llm.invoke(messages)
        answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        answer = f"產生答案時發生錯誤：{e}"

    await send_text(answer)
    await send_text(list_files_text())      # 回清單形成循環

# ---------- Telegram Handler ----------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    # /start 或 /files：列清單
    if msg.text and msg.text.strip() in ("/start", "/files"):
        await msg.reply_text(list_files_text())
        return

    # 小幫手：送出照片（必要時用；目前回答不自動回圖）
    async def _send_photo(path):
        try:
            with open(path, "rb") as f:
                await msg.reply_photo(photo=f)
        except Exception as e:
            print("reply_photo 失敗：", e)

    # ---------- PDF ----------
    if msg.document and msg.document.mime_type == "application/pdf":
        tg_file = await context.bot.get_file(msg.document.file_id)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        await tg_file.download_to_drive(tmp.name)
        tmp.close()
        fid = await process_pdf(tmp.name)
        await msg.reply_text(f"✅ PDF 已儲存並處理（檔案 ID：{fid}）\n\n" + list_files_text())
        return

    # ---------- Photo ----------
    if msg.photo:
        tg_file = await context.bot.get_file(msg.photo[-1].file_id)
        bio = BytesIO()
        await tg_file.download_to_memory(out=bio)
        img_bytes = bio.getvalue()
        fid = await process_photo(img_bytes, f"{msg.photo[-1].file_id}.jpg")
        await msg.reply_text(f"✅ 圖片已處理並寫入資料庫（檔案 ID：{fid}）\n\n" + list_files_text())
        return

    # ---------- Text ----------
    if msg.text:
        text = msg.text.strip()
        m = ID_QUERY_RE.match(text)
        if not m:
            await msg.reply_text(
                "請以『ID: 問題』格式查詢，例如：\n\n"
                "`12: 請幫我摘要這份手冊的安全注意事項`\n\n" +
                list_files_text()
            )
            return
        fid = int(m.group(1))
        q = m.group(2).strip()
        await qa_with_file_scope(fid, q, msg.reply_text)
        return

# ---------- Bot 啟動 -------------------------------------------------------
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_message))
    app.add_handler(CommandHandler("files", handle_message))
    app.add_handler(MessageHandler(filters.ALL, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
