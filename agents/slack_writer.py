from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage, AIMessage
from langchain_core.tools.structured import StructuredTool
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import operator
import os
import subprocess
from dotenv import load_dotenv
load_dotenv()

# model config
model = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1",
  model="anthropic/claude-3.5-sonnet"
)

# define tools
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

def send_slack_message(recipient: str, message: str) -> str:
    """
    Sends a message to a Slack user or channel.
    You can pass a user ID, username (@user_name), or channel name (#channel_name).
    """
    try:
        # Note: Slack requires '#channel' syntax for public/private channels
        response = client.chat_postMessage(
            channel=recipient,
            text=message
        )
        return f"Message sent successfully to {recipient}."
    except SlackApiError as e:
        # Return the specific error message to the LLM
        return f"Error sending message: {e.response['error']}"
    

def run_main():
    root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    main_path = os.path.join(root_path, "main.py")

    if not os.path.isfile(main_path):
        return f"main.py not found at expected path: {main_path}"

    try:
        result = subprocess.run(
            ["python", main_path],
            check=True,
            capture_output=True,
            text=True
        )
        return f"Output:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error while running main.py:\n{e.stderr}"

    
send_message_tool = StructuredTool.from_function(
    func=send_slack_message,
    name="send_slack_message",
    description="Send a Slack message to a specific user or channel. The recipient must be a valid user ID, username (e.g., @john_doe), or channel name (e.g., #general)."
)

start_process_tool = StructuredTool.from_function(
    func=run_main,
    name="start_process",
    description="Runs the process that will split a plan with user stories into tasks"
)

tools = [send_message_tool, start_process_tool]

# Bind the tool for tool calling capability
llm_with_tools = model.bind_tools(tools)


# --- 1. Define the State Schema ---
# We use operator.add to append new messages to the list,
# which is essential for the conversational history in the agent loop.

class SlackWriterState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# --- 4. Define Agent Nodes ---

def llm_call(state: SlackWriterState) -> SlackWriterState:
    """The LLM node: calls the model to decide on an action."""
    system_prompt = (
        "You are an automated Slack assistant tasked with sending messages and starting the planngin process. "
        "If the user explicitly asks you to 'send' a message, you MUST call "
        "the 'send_slack_message' tool. If the user's request is a greeting "
        "or a question not related to sending a message, respond conversationally "
        "and DO NOT use the tool. Always be concise."
        "If the user gives you a plan or user stories, use the 'start_process' tool"
        "Update the user after calling the start process tool, with the 'send_slack_message' tool"
    )
    
    # Prepend the system prompt to the messages list
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]} # Add the LLM's response

def tool_node(state: SlackWriterState) -> SlackWriterState:
    """The Tool node: executes the tool call requested by the LLM."""
    last_message = state["messages"][-1]
    tool_messages = []
    
    # Check for tool calls and execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == send_message_tool.name:
                # Execute the specific tool
                observation = send_slack_message(**tool_args)
                
                # Append the result as a ToolMessage
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_call_id
                    )
                )

            if tool_name == start_process_tool.name:
                # Execute the specific tool
                observation = run_main(**tool_args)
                
                # Append the result as a ToolMessage
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_call_id
                    )
                )
    
    # Return the observation(s) from the tool execution
    return {"messages": tool_messages}

# --- 5. Define Conditional Edge Function ---

def route_decision(state: SlackWriterState) -> Literal["tool_node", END]:
    """Decides the next step: either call the tool or end the graph."""
    last_message = state["messages"][-1]
    
    # If the LLM requested a tool call, route to the 'tool_node' node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    
    # Otherwise, the LLM has generated a final answer, so end the graph
    return END

# --- 6. Build and Compile Graph ---

# Initialize the StateGraph with the defined state schema
graph_builder = StateGraph(SlackWriterState)

# Add nodes
graph_builder.add_node("llm_call", llm_call)
graph_builder.add_node("tool_node", tool_node)

# Set the entry transition: from START to the LLM node
graph_builder.add_edge(START, "llm_call")

# Set the conditional transition from the LLM node
graph_builder.add_conditional_edges(
    "llm_call",
    route_decision,
    {"tool_node": "tool_node", END: END}
)

# Set the transition from the Tool node back to the LLM (for reflection/final answer)
graph_builder.add_edge("tool_node", "llm_call")

# Compile the graph into a runnable agent
agent = graph_builder.compile()

def run_agent(user_input):
    initial_message = [HumanMessage(content=user_input)]
    print(f"--- Running Agent for: '{user_input}' ---")

    result = agent.invoke({"messages": initial_message})
    
    final_message = result["messages"][-1].content
    print("\nFinal Agent Response:")
    print(f"=============================\n{final_message}")

    return final_message

# Example Run
run_agent("send 'Hello from the Pacer Agent!' to #all-agentsverse-hackathon")


# --- 7. Run the Agent (Example) ---
# user_input = "Send 'Hello from the LangGraph Agent!' to #all-agentsverse-hackathon"
# user_input = "What is the capital of France?" # Example without tool call

# initial_message = [HumanMessage(content=user_input)]
# print(f"--- Running Agent for: '{user_input}' ---")

# The agent's state now starts with the user's initial message
# result = agent.invoke({"messages": initial_message})

# Print the final output message from the agent
# final_message = result["messages"][-1].content
# print("\nFinal Agent Response:")
# print(f"=============================\n{final_message}")

# You can visualize the graph if you have the necessary libraries installed
# from IPython.display import Image, display
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))