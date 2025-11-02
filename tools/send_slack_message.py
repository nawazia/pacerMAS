from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_core.tools.structured import StructuredTool
import os

client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

def send_slack_message(recipient: str, message: str) -> str:
    """
    Sends a message to a Slack user or channel.
    You can pass a user ID, username (with '@'), or channel name.
    """
    try:
        response = client.chat_postMessage(
            channel=recipient,
            text=message
        )
        return f"✅ Message sent to {recipient}"
    except SlackApiError as e:
        return f"❌ Slack API error: {e.response['error']}"
    

# Define the tool
send_message_tool = StructuredTool.from_function(
    func=send_slack_message,
    name="send_slack_message",
    description="Send a Slack message to a specific user or channel. Takes recipient and message as arguments."
)
