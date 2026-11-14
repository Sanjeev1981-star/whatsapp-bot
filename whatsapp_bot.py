"""
WhatsApp auto-reply assistant
Meta Cloud API + Flask webhook + LLM (Gemini free tier by default, Claude optional)

Setup:
  pip install flask requests
  export WA_TOKEN=...            # Meta permanent/temporary access token
  export WA_PHONE_ID=...         # Phone number ID from Meta dashboard
  export VERIFY_TOKEN=...        # any string you choose; paste same in Meta webhook setup
  export LLM_PROVIDER=gemini     # or: claude
  export GEMINI_API_KEY=...      # if gemini
  export GEMINI_MODEL=gemini-flash-latest,gemini-3.1-flash-lite   # tried in order
  export ANTHROPIC_API_KEY=...   # if claude
  export CLAUDE_MODEL=claude-sonnet-4-6
  python whatsapp_bot.py
Then expose it (e.g. ngrok http 5000) and set webhook URL to https://<host>/webhook
"""
import os
import time
from collections import defaultdict, deque

import requests
from flask import Flask, request

app = Flask(__name__)

WA_TOKEN = os.environ.get("WA_TOKEN", "")
WA_PHONE_ID = os.environ.get("WA_PHONE_ID", "")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "change-me")
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")

SYSTEM_PROMPT = (
    "You are a helpful personal WhatsApp assistant. Reply briefly and clearly. "
    "Reply in the same language the sender wrote in."
)

# Short per-sender memory (last 10 turns), in RAM only
history = defaultdict(lambda: deque(maxlen=10))
seen_ids = deque(maxlen=500)  # avoid double replies on webhook retries


def _gemini_call(model: str, turns: list) -> str:
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},  # key kept out of the URL/logs
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user" if t["role"] == "user" else "model",
                    "parts": [{"text": t["text"]}],
                }
                for t in turns
            ],
        },
        timeout=25,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask_llm(user_id: str, text: str) -> str:
    history[user_id].append({"role": "user", "text": text})
    turns = list(history[user_id])
    try:
        if PROVIDER == "claude":
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
                    "max_tokens": 500,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": t["role"], "content": t["text"]} for t in turns
                    ],
                },
                timeout=30,
            )
            r.raise_for_status()
            reply = r.json()["content"][0]["text"]
        else:
            # Try each model in order; retry briefly on overload (429/500/503)
            models = [
                m.strip()
                for m in os.environ.get(
                    "GEMINI_MODEL", "gemini-flash-latest,gemini-3.1-flash-lite"
                ).split(",")
                if m.strip()
            ]
            reply, last_err = None, None
            for model in models:
                for attempt in range(2):
                    try:
                        reply = _gemini_call(model, turns)
                        break
                    except requests.HTTPError as e:
                        last_err = f"{model}: HTTP {e.response.status_code}"
                        if e.response.status_code in (429, 500, 503):
                            time.sleep(2)
                            continue
                        break  # 400/403/404: try the next model
                if reply:
                    break
            if not reply:
                raise RuntimeError(last_err or "no reply")
    except Exception:
        history[user_id].pop()  # don't keep an unanswered turn
        raise

    history[user_id].append({"role": "assistant", "text": reply})
    return reply


def send_whatsapp(to: str, text: str) -> None:
    requests.post(
        f"https://graph.facebook.com/v21.0/{WA_PHONE_ID}/messages",
        headers={"Authorization": f"Bearer {WA_TOKEN}"},
        json={
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": text[:4000]},
        },
        timeout=30,
    )


@app.get("/webhook")
def verify():
    if (
        request.args.get("hub.mode") == "subscribe"
        and request.args.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return request.args.get("hub.challenge", ""), 200
    return "forbidden", 403


@app.post("/webhook")
def incoming():
    data = request.get_json(silent=True) or {}
    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                for msg in change.get("value", {}).get("messages", []):
                    if msg["id"] in seen_ids:
                        continue
                    seen_ids.append(msg["id"])
                    sender = msg["from"]
                    if msg.get("type") != "text":
                        send_whatsapp(sender, "I can only read text messages for now.")
                        continue
                    try:
                        reply = ask_llm(sender, msg["text"]["body"])
                    except Exception as e:  # LLM failure shouldn't break webhook
                        print("LLM error:", e)
                        reply = "Sorry, I couldn't process that. Please try again."
                    send_whatsapp(sender, reply)
    except Exception as e:
        print("Webhook error:", e)
    return "ok", 200  # always 200 so Meta doesn't retry endlessly


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)))
