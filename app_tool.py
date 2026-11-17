# -----------------------------
# IMPORTS
# -----------------------------

import streamlit as st
import uuid

from langchain_core.messages import HumanMessage, AIMessageChunk

from DotAi_with_tools import workflow, get_all_threads


# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="DotAI",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# SESSION STATE
# -----------------------------

# Remember which conversation the user is currently viewing
if "active_chat_id" not in st.session_state:

    st.session_state.active_chat_id = None


# -----------------------------
# CREATE NEW CHAT
# -----------------------------

def create_new_chat():

    # Every new conversation gets a unique ID
    new_chat_id = str(uuid.uuid4())

    st.session_state.active_chat_id = new_chat_id


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.title("🤖 DotAI")

    # Start a fresh conversation
    if st.button("➕ New Chat", use_container_width=True):

        create_new_chat()

        st.rerun()

    st.divider()

    st.subheader("Your Chats")

    # Get all conversations saved in SQLite
    all_threads = get_all_threads()

    # Display previously saved conversations
    for thread_id in reversed(all_threads):

        if st.button(
            f"Chat {thread_id[:8]}",
            key=thread_id,
            use_container_width=True
        ):

            st.session_state.active_chat_id = thread_id

            st.rerun()


# -----------------------------
# MAIN CHAT AREA
# -----------------------------

st.title("DotAI Chatbot")


# Automatically create a chat when the application first opens
if st.session_state.active_chat_id is None:

    create_new_chat()


current_chat_id = st.session_state.active_chat_id


# LangGraph uses this configuration to identify the conversation
CONFIG = {
    "configurable": {
        "thread_id": current_chat_id
    },

    "metadata": {
        "thread_id": current_chat_id
    },

    "run_name": "chat_trace"
}


# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------

# Retrieve the saved conversation from SQLite
state = workflow.get_state(CONFIG)

messages = (
    state.values.get("messages", [])
    if state.values
    else []
)


# Display previous messages in the chat window
for message in messages:

    if message.type == "human":

        role = "user"

    elif message.type == "ai" and message.content:

        role = "assistant"

    else:

        # Don't display internal tool messages
        continue

    with st.chat_message(role):

        st.markdown(message.content)


# -----------------------------
# USER INPUT
# -----------------------------

user_input = st.chat_input("Ask DotAI anything...")


if user_input:

    # Display the user's message
    with st.chat_message("user"):

        st.markdown(user_input)


    # Display the AI response as it is generated
    with st.chat_message("assistant"):

        response_area = st.empty()

        response_text = ""


        # Run the LangGraph workflow and receive streaming messages
        for message_chunk, metadata in workflow.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },

            config=CONFIG,

            stream_mode="messages"
        ):

            # Only display AI-generated text, not raw tool results
            if isinstance(message_chunk, AIMessageChunk):

                if isinstance(message_chunk.content, str):

                    response_text += message_chunk.content

                    response_area.markdown(response_text + "▌")


        # Display the completed response
        if response_text:

            response_area.markdown(response_text)

        else:

            response_area.write("I couldn't generate a response. Please try again.")


    # Refresh the page to load the updated conversation history
    st.rerun()