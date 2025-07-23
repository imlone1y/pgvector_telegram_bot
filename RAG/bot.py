from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import os, tempfile, fitz, pytesseract, psycopg2
from PIL import Image
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import json

load_dotenv()

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
PG_CONF = {
    "host": os.environ["PG_HOST"],
    "port": os.environ["PG_PORT"],
    "dbname": os.environ["PG_DB"],
    "user": os.environ["PG_USER"],
    "password": os.environ["PG_PASSWORD"]
}

# PDF 處理函式
async def process_pdf_and_store(file_path):
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    documents = splitter.split_documents(raw_documents)

    ocr_texts = []
    pdf_name = os.path.basename(file_path)
    pdf_doc = fitz.open(file_path)

    conn = psycopg2.connect(**PG_CONF)
    cur = conn.cursor()

    # 先處理純文字段落
    for doc in documents:
        content = doc.page_content
        vector = embedding_model.embed_query(content)
        vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
        cur.execute("""
            INSERT INTO documents (content, embedding, source_type, filename, page_num)
            VALUES (%s, %s, %s, %s, %s)
        """, (content, vector_str, 'pdf_text', pdf_name, None))

    # 再處理圖片 OCR
    for page_index in range(len(pdf_doc)):
        images = pdf_doc[page_index].get_images(full=True)
        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]

            image_filename = f"{pdf_name}_page{page_index+1}_img{img_index+1}.png"
            image_path = os.path.join("image_dir", image_filename)
            os.makedirs("image_dir", exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang="chi_tra+eng")

            if text.strip():
                vector = embedding_model.embed_query(text)
                vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
                cur.execute("""
                    INSERT INTO documents (content, embedding, source_type, filename, page_num, image_ref, image_desc)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (text, vector_str, 'ocr_image', pdf_name, page_index+1, image_filename, None))


    conn.commit()
    cur.close()
    conn.close()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    text = message.text.strip() if message.text else ""

    if message.text == "/start":
        await message.reply_text("初始化完成，可以開始使用")

    elif message.document and message.document.mime_type == 'application/pdf':
        file = await context.bot.get_file(message.document.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            await file.download_to_drive(custom_path=tmp.name)
            await process_pdf_and_store(tmp.name)
        await message.reply_text("✅ PDF 內容與圖片（OCR）已上傳並寫入資料庫")

    elif message.photo:
        file = await context.bot.get_file(message.photo[-1].file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            await file.download_to_drive(custom_path=tmp.name)

            image_filename = os.path.basename(tmp.name)
            image_path = os.path.join("image_dir", image_filename)
            os.makedirs("image_dir", exist_ok=True)
            os.rename(tmp.name, image_path)

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang="chi_tra+eng").strip()

            if text:
                vector = embedding_model.embed_query(text)
                vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"

                conn = psycopg2.connect(**PG_CONF)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO documents (content, embedding, source_type, image_ref, filename, page_num)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (text, vector_str, 'uploaded_image', image_filename, image_filename, None))
                conn.commit()
                cur.close()
                conn.close()

                await message.reply_text("✅ 已接收圖片並完成 OCR 與嵌入")
            else:
                await message.reply_text("無法從圖片中辨識出文字，未寫入資料庫")

    elif message.text:
        user_message = message.text.strip()

        # --- STEP 1: 使用 LLM 判斷是否為修改指令 ---
        system_prompt = f"""
        你是一個幫助使用者修改資料庫中文本內容的助手。
        請判斷以下訊息是否要修改資料：

        「{user_message}」

        如果是，請用 JSON 格式回傳：
        {{"action": "modify", "old_text": "舊的內容", "new_text": "新的內容"}}

        如果不是，請回傳：
        {{"action": "none"}}
        """

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        result = llm.invoke([{"role": "system", "content": system_prompt}]).content

        try:
            data = json.loads(result)
            if data.get("action") == "modify":
                old_text = data["old_text"]
                new_text = data["new_text"]

                old_vec = embedding_model.embed_query(old_text)
                old_vec_str = "[" + ",".join(f"{x:.8f}" for x in old_vec) + "]"

                conn = psycopg2.connect(**PG_CONF)
                cur = conn.cursor()
                cur.execute("SELECT id FROM documents ORDER BY embedding <-> %s LIMIT 1", (old_vec_str,))
                found = cur.fetchone()

                if found:
                    doc_id = found[0]
                    new_vec = embedding_model.embed_query(new_text)
                    new_vec_str = "[" + ",".join(f"{x:.8f}" for x in new_vec) + "]"

                    cur.execute("UPDATE documents SET content=%s, embedding=%s WHERE id=%s",
                                (new_text, new_vec_str, doc_id))
                    conn.commit()
                    await message.reply_text(f"✅ 已將「{old_text}」更新為「{new_text}」")
                else:
                    await message.reply_text("找不到要修改的內容，請確認原本內容是否存在")

                cur.close()
                conn.close()
                return  # 不進行後續問答

        except Exception as e:
            print("判斷修改語句失敗：", e)

        # --- STEP 2: 正常問答流程 ---
        question = user_message
        vector = embedding_model.embed_query(question)
        vector_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"

        conn = psycopg2.connect(**PG_CONF)
        cur = conn.cursor()
        cur.execute("""
            SELECT content, image_ref, image_desc FROM documents
            ORDER BY embedding <-> %s
            LIMIT 3;
        """, (vector_str,))
        results = cur.fetchall()
        context_text = "\n".join(r[0] for r in results if r[0])
        system_prompt = f"You are a helpful assistant. Use the following context to answer:\n{context_text}"

        answer = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]).content

        await message.reply_text(answer)

        for content, image_ref, image_desc in results:
            if image_ref:
                image_path = os.path.join("image_dir", image_ref)
                if os.path.exists(image_path):
                    await context.bot.send_photo(chat_id=message.chat_id, photo=open(image_path, "rb"))
                break



app = ApplicationBuilder().token(bot_token).build()
app.add_handler(MessageHandler(filters.ALL, handle_message))
app.run_polling()
