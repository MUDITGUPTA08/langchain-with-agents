from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from tools import calculator, word_counter, unit_converter, get_current_datetime, text_analyzer


# Load API Key (supports both .env locally and Streamlit Cloud secrets)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")


# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# Tools
tools = [calculator, word_counter, unit_converter, get_current_datetime, text_analyzer]


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful AI assistant with access to several tools.\n"
         "Use tools when they are relevant to the user's request.\n"
         "- Use calculator for math expressions.\n"
         "- Use word_counter for counting words/characters.\n"
         "- Use unit_converter for converting units (km to miles, etc).\n"
         "- Use get_current_datetime when asked about the current date or time.\n"
         "- Use text_analyzer to analyze text statistics.\n"
         "For everything else, answer from your own knowledge."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_agent(query, chat_history):
    result = agent_executor.invoke({"input": query, "chat_history": chat_history})
    return result["output"]


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Gemini Agent", page_icon="🤖", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🤖 Gemini Agent")
    st.markdown("Powered by **Gemini 2.5 Flash** + LangChain")
    st.divider()

    st.subheader("🛠️ Available Tools")
    tool_info = {
        "🧮 Calculator": "Solves math expressions\n`e.g. 2**10, math.sqrt(16)`",
        "📝 Word Counter": "Counts words, chars & sentences",
        "📐 Unit Converter": "Converts units\n`e.g. 5 km to miles`",
        "🕐 Date & Time": "Returns current date and time",
        "🔍 Text Analyzer": "Analyzes word frequency & stats",
    }
    for tool_name, desc in tool_info.items():
        with st.expander(tool_name):
            st.caption(desc)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with LangChain + Streamlit")


# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Page header
st.title("💬 Chat with Gemini Agent")
st.caption("Ask me anything — I can do math, convert units, analyze text, and more!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask something..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Build chat history for context (last 10 messages)
    history = []
    for msg in st.session_state.messages[-10:]:
        if msg["role"] == "user":
            history.append(("human", msg["content"]))
        else:
            history.append(("ai", msg["content"]))

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_agent(query, history)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
