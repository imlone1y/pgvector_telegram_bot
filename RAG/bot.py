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
llm = ChatOpenAI(model="gpt-5", temperature=1)

STRICT_SYS_PROMPT = """1. 只引用來源能支持的內容；不可外部常識延伸。
2. 每一句話後面都必須加上 [doc_id:段落或頁碼] 引用；若沒有足夠來源，就不回答。
3. 禁止使用來源未出現的數字、名詞定義與結論。
4. 禁止改寫成與原意矛盾的說法；對專有名詞保留原文。"""


PG_CONF = dict(
    host=os.environ["PG_HOST"],
    port=os.environ["PG_PORT"],
    dbname=os.environ["PG_DB"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"]
)

# ---- 相似度門檻（cosine distance；越小越像） -------------------------------
IMG_SIM_THRESHOLD = float(os.getenv("IMG_SIM_THRESHOLD", "0.75"))
IMG_TOPK = int(os.getenv("IMG_TOPK", "5"))

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

# ---------- 文字問答 / 修改（只在相符時回圖；不回 OCR 文字） -----------------
async def qa_or_modify(user_msg: str, send_text, send_photo):
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
                    await send_text(f"✅ 已將「{j['old_text']}」更新為「{j['new_text']}」")
                    return
            await send_text("找不到要修改的內容，請確認原始內容是否存在")
            return
    except Exception as e:
        print("判斷修改語句失敗：", e)

    # 2) 問答流程
    vec = embedding_model.embed_query(user_msg)
    vec_str = vector_to_str(vec)
    
    # 2a) 取前 3 筆相似內容組成 context（含來源資訊）
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT content, filename, page_num, source_type
            FROM documents
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> %s
            LIMIT 3
        """, (vec_str,))
        top = cur.fetchall()

    context_parts = []
    sources = []
    for content, filename, page_num, source_type in top:
        if content:
            context_parts.append(content)
            src = f"{filename}"
            if page_num: src += f" p.{page_num}"
            if source_type: src += f" ({source_type})"
            sources.append(src)

    context = "\n".join(context_parts)

    # 嚴格只用 Context 回答
    if not context.strip():
        await send_text("我不知道。無法在你提供的資料中找到足夠資訊，請上傳或提供更多相關內容。")
        return

    answer = llm.invoke([
        {"role": "system", "content": STRICT_SYS_PROMPT},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user",   "content": user_msg}
    ]).content


    # 回覆答案 + 資料來源
    if sources:
        answer += "\n\n📖 資料來源:\n" + "\n".join(f"- {s}" for s in sources)

    await send_text(answer)

    # 2b) 只有在相似度達標時才回傳圖片（不含任何文字）
    #     使用 cosine distance `<=>`（需 pgvector 支援；未建索引也能執行，但較慢）
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                SELECT image_ref, (embedding <=> %s) AS dist
                FROM documents
                WHERE image_ref IS NOT NULL
                  AND embedding IS NOT NULL
                ORDER BY dist ASC
                LIMIT {IMG_TOPK}
            """, (vec_str,))
            imgs = cur.fetchall()

        for image_ref, dist in imgs:
            if image_ref and dist is not None and dist <= IMG_SIM_THRESHOLD:
                img_path = os.path.join(IMAGE_DIR, image_ref)
                if os.path.exists(img_path):
                    await send_photo(img_path)   # ← 不傳 caption，只傳圖片
                    break  # 只回第一張達標圖
    except Exception as e:
        print("回傳圖片流程錯誤：", e)

# ---------- Telegram Handler ----------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg: return

    if msg.text and msg.text.strip() == "/start":
        await msg.reply_text("初始化完成，可以開始使用")
        return

    # 小幫手：送圖（確保檔案關閉；不帶 caption）
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
        await qa_or_modify(msg.text.strip(), msg.reply_text, _send_photo)
        return

# ---------- Bot 啟動 -------------------------------------------------------
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.ALL, handle_message))
app.run_polling()
