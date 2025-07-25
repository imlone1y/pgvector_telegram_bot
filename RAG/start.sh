#!/bin/bash
# 啟動 Telegram Bot
python bot.py &

# 啟動 Streamlit 並綁定到 0.0.0.0:8080（Zeabur 要這樣才能連外）
streamlit run image.py --server.port 8080 --server.address 0.0.0.0