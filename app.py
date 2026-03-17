"""
This chat example using MCPAgent with built-in conversation memory.

The example demonstrates how to use the MCPAgent with the built in 
Conversation History capabilities for better contextual interactions.
"""

import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
import os
import sys

# Prevent Windows console encoding errors when logs contain Unicode (e.g., emojis).
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

async def run_memory_chat():
    """ Run a chat using MCPAgents built in conversation history."""

    load_dotenv()

    # Use GROQ API Key using https://console.groq.com/keys
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    config_file = "browser_mcp.json"

    # Create MCPClient and agent with memory enabled
    client = MCPClient(config=config_file)
    # Set GROQ_MODEL to override (e.g. "qwen/qwen3-32b", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    # Create Agent with memory enabled=True
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,
        additional_instructions=(
            "When calling tools, strictly follow each tool's JSON schema. "
            "Use correct JSON types (e.g., numbers must be numbers, not strings). "
            "Only include parameters that the schema allows."
        ),
    )

    print("\n===========Interactive MCP Chat============")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear the conversation history")
    print("==============================================")

    try:
        # Main chat loop
        while True:
            # Get user input
            user_input = input("\nYou:")

            # Exit if user wants to quit
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break

            # Clear conversation history if user wants to clear
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("\nConversation history cleared.")
                continue

            # Get response from agent
            print("\nAssistant: ", end="", flush=True)

            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"Error: {e}")
    finally:
        # Clean up
        if getattr(client, "sessions", None):
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())



