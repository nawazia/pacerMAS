from fastapi import FastAPI, Request, BackgroundTasks
import requests
import os
import time
from dotenv import load_dotenv
from pathlib import Path
from agents.slack_writer import run_agent
from integrations.slack_listener.slack_utils import normalize_slack_text


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

app = FastAPI()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

MAX_EVENT_AGE_SECONDS = 30

@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    if data.get("type") == "url_verification":
        return {"challenge": data["challenge"]}

    event_time = data.get("event_time", 0)
    now = int(time.time())

    # Ignore stale events (sent before the app started or retries)
    if now - event_time > MAX_EVENT_AGE_SECONDS:
        print(f"⚠️ Ignoring stale event (age {now - event_time}s)")
        return {"ok": True}


    # Handle verification challenge
    if data.get("type") == "url_verification":
        return {"challenge": data["challenge"]}

    # Only process message events
    if "event" in data:
        event = data["event"]

        # Ignore bot messages
        if event.get("subtype") == "bot_message" or event.get("bot_id"):
            return {"ok": True}

        user = event.get("user")
        text = event.get("text")
        channel = event.get("channel")

        print(f"Received message from {user}: {text}")

        # 👉 Process in the background to avoid Slack retrying
        background_tasks.add_task(process_message, user, text, channel)

    # Immediately acknowledge
    return {"ok": True}


def process_message(user, text, channel):
    """Runs the agent and sends the response to Slack."""
    
    clean_text = normalize_slack_text(text)

    try:
        reply_text = run_agent(clean_text)
    except Exception as e:
        reply_text = f"Error running agent: {e}"

    requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        json={"channel": channel, "text": reply_text}
    )
