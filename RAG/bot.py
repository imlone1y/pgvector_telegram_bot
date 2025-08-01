"""
Telegram Bot － 對齊新版 documents 資料表
columns: id, page_num, content, embedding, created_at,
         image_ref, image_desc, source_type, filename,
         image_hash, upload_time
"""

import os, uuid, hashlib, datetime, tempfile, json
from io import BytesIO
import psycopg2, fitz, pytesseract
from PIL import Image
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ---------- 基礎設定 --------------------------------------------------------
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
IMAGE_DIR  = "image_dir"; os.makedirs(IMAGE_DIR, exist_ok=True)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

PG_CONF = dict(
    host=os.environ["PG_HOST"],
    port=os.environ["PG_PORT"],
    dbname=os.environ["PG_DB"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"]
)

# ---------- 共用小函式 ------------------------------------------------------
def db_conn():
    return psycopg2.connect(**PG_CONF)

def vector_to_str(vec):
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

def insert_pdf_text(cur, content, filename):
    vec_str = vector_to_str(embedding_model.embed_query(content))
    cur.execute("""
        INSERT INTO documents (content, embedding, source_type,
                               filename, upload_time)
        VALUES (%s,%s,'pdf_text',%s,%s)
    """, (content, vec_str, filename, datetime.datetime.utcnow()))

def save_image_and_insert(cur, img_bytes, ocr_text,
                          source_type, filename, page_num=None):
    # 計算 hash，若重複就略過
    img_hash = hashlib.md5(img_bytes).hexdigest()
    cur.execute("SELECT 1 FROM documents WHERE image_hash=%s", (img_hash,))
    if cur.fetchone():
        return

    image_ref = f"{uuid.uuid4().hex}_{filename}"
    img_path  = os.path.join(IMAGE_DIR, image_ref)
    with open(img_path, "wb") as f: f.write(img_bytes)

    vec_str = None
    if ocr_text:
        vec_str = vector_to_str(embedding_model.embed_query(ocr_text))

    cur.execute("""
        INSERT INTO documents (content, embedding, source_type,
                               filename, page_num,
                               image_ref, image_hash, upload_time)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (ocr_text or None, vec_str, source_type,
          filename, page_num,
          image_ref, img_hash, datetime.datetime.utcnow()))

# ---------- PDF 上傳處理 ----------------------------------------------------
async def process_pdf(file_path):
    filename  = os.path.basename(file_path)
    loader    = PyPDFLoader(file_path)
    pdf_doc   = fitz.open(file_path)
    splitter  = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    with db_conn() as conn:
        with conn.cursor() as cur:
            # 文字分段
            for doc in splitter.split_documents(loader.load()):
                if doc.page_content.strip():
                    insert_pdf_text(cur, doc.page_content.strip(), filename)

            # 圖片 OCR
            for p in range(len(pdf_doc)):
                for xref, *_ in pdf_doc[p].get_images(full=True):
                    img_bytes = pdf_doc.extract_image(xref)["image"]
                    ocr_text  = pytesseract.image_to_string(
                        Image.open(BytesIO(img_bytes)),
                        lang="chi_tra+eng"
                    ).strip()
                    save_image_and_insert(cur, img_bytes, ocr_text,
                                          'ocr_image', filename, page_num=p+1)
        conn.commit()

# ---------- 圖片上傳處理 ----------------------------------------------------
async def process_photo(img_bytes, original_name):
    ocr_text = pytesseract.image_to_string(
        Image.open(BytesIO(img_bytes)),
        lang="chi_tra+eng"
    ).strip()
    with db_conn() as conn:
        with conn.cursor() as cur:
            save_image_and_insert(cur, img_bytes, ocr_text,
                                  'uploaded_image', original_name)
        conn.commit()

# ---------- 文字問答 / 修改 -------------------------------------------------
async def qa_or_modify(user_msg: str, bot_send):
    # 1) 判斷是否為「修改」指令
    sys_prompt = f"""
你是一個幫助使用者修改資料庫內容的助手。
請判斷以下訊息是否要修改資料：
「{user_msg}」
若需修改，回傳：
{{"action":"modify","old_text":"...","new_text":"..."}}
否則回傳：{{"action":"none"}}
"""
    try:
        j = json.loads(llm.invoke([{"role":"system","content":sys_prompt}]).content)
        if j.get("action") == "modify":
            old_vec_str = vector_to_str(embedding_model.embed_query(j["old_text"]))
            with db_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT id FROM documents ORDER BY embedding <-> %s LIMIT 1",
                            (old_vec_str,))
                row = cur.fetchone()
                if row:
                    new_vec_str = vector_to_str(embedding_model.embed_query(j["new_text"]))
                    cur.execute("""UPDATE documents
                                   SET content=%s, embedding=%s
                                   WHERE id=%s""",
                                (j["new_text"], new_vec_str, row[0]))
                    conn.commit()
                    await bot_send(f"✅ 已將「{j['old_text']}」更新為「{j['new_text']}」")
                    return
            await bot_send("找不到要修改的內容，請確認原始內容是否存在")
            return
    except Exception as e:
        print("判斷修改語句失敗：", e)

    # 2) 問答流程
    vec_str = vector_to_str(embedding_model.embed_query(user_msg))
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT content, image_ref
            FROM documents
            ORDER BY embedding <-> %s
            LIMIT 3
        """, (vec_str,))
        top = cur.fetchall()

    context = "\n".join(t[0] for t in top if t[0])
    answer = llm.invoke([
        {"role":"system","content":f"Use this context:\n{context}"},
        {"role":"user","content":user_msg}
    ]).content
    await bot_send(answer)

# ---------- Telegram Handler ----------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg: return

    if msg.text and msg.text.strip() == "/start":
        await msg.reply_text("初始化完成，可以開始使用")
        return

    # ---------- PDF ----------
    if msg.document and msg.document.mime_type == "application/pdf":
        tg_file = await context.bot.get_file(msg.document.file_id)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        await tg_file.download_to_drive(tmp.name)
        tmp.close()
        await process_pdf(tmp.name)
        await msg.reply_text("✅ PDF 已儲存並處理")
        return

    # ---------- Photo ----------
    if msg.photo:
        tg_file = await context.bot.get_file(msg.photo[-1].file_id)
        bio = BytesIO()
        await tg_file.download_to_memory(out=bio)
        img_bytes = bio.getvalue()
        await process_photo(img_bytes, f"{msg.photo[-1].file_id}.jpg")
        await msg.reply_text("✅ 圖片已處理並寫入資料庫")
        return

    # ---------- Text ----------
    if msg.text:
        await qa_or_modify(msg.text.strip(), msg.reply_text)
        return

# ---------- Bot 啟動 -------------------------------------------------------
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.ALL, handle_message))
app.run_polling()
