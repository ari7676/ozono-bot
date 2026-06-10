from flask import Flask, request
import requests
import threading
import time

app = Flask(__name__)

TELEGRAM_TOKEN = "8653836924:AAH8w4wUXU_wdAUb3e2ZDmb5xKh0o7F5WEk"
CHAT_ID = "512404508"

def keep_alive():
    while True:
        time.sleep(600)
        try:
            requests.get("https://trading-alerts-1blr.onrender.com/")
        except:
            pass

threading.Thread(target=keep_alive, daemon=True).start()

@app.route("/", methods=["GET"])
def home():
    return {"ok": True}, 200

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True)
    
    if data and "value" in data:
        message = data["value"]
    elif request.data:
        message = request.data.decode("utf-8")
    else:
        return {"ok": False, "error": "empty"}, 400

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": message})
    
    return {"ok": True}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
