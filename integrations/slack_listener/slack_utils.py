import requests
import os
import re

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

def get_channel_name_from_id(channel_id: str) -> str:
    """Return the human-readable channel name for a given Slack channel ID."""
    response = requests.get(
        "https://slack.com/api/conversations.info",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        params={"channel": channel_id}
    )
    data = response.json()
    if data.get("ok"):
        name = data["channel"]["name"]
        # prefix with "#" to look natural in messages
        return f"#{name}"
    else:
        print(f"⚠️ Could not resolve channel {channel_id}: {data.get('error')}")
        return f"#{channel_id}"

def normalize_slack_text(text: str) -> str:
    """Replace Slack’s <#C123|> etc. with real names using the API."""
    if not text:
        return ""

    # Replace channel mentions like <#C123|general> or <#C123|>
    def replace_channel(match):
        channel_id = match.group(1)
        visible_name = match.group(2)
        if visible_name:  # if Slack provided a name
            return f"#{visible_name}"
        # otherwise, fetch via API
        return get_channel_name_from_id(channel_id)

    text = re.sub(r"<#([A-Z0-9]+)\|([^>]*)>", replace_channel, text)

    # Mentions: <@U123|john> → @john
    text = re.sub(r"<@([A-Z0-9]+)\|([^>]*)>", lambda m: f"@{m.group(2) or m.group(1)}", text)

    # Links: <https://url|label> → label
    text = re.sub(r"<https?://[^|]+\|([^>]+)>", r"\1", text)
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)

    return text.strip()
