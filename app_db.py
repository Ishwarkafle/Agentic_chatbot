import streamlit as st
import uuid

from langchain_core.messages import HumanMessage

from DotAi_DB import workflow, get_all_threads


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

# Store only the currently selected chat ID
if "active_chat_id" not in st.session_state:

    st.session_state.active_chat_id = None


# -----------------------------
# CREATE NEW CHAT
# -----------------------------

def create_new_chat():

    new_chat_id = str(uuid.uuid4())

    st.session_state.active_chat_id = new_chat_id


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.title("🤖 DotAI")

    if st.button("➕ New Chat", use_container_width=True):

        create_new_chat()
        st.rerun()

    st.divider()

    st.subheader("Your Chats")

    # Get saved chats from SQLite
    all_threads = get_all_threads()

    # Show newest chats first
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


# Create a chat automatically if none is selected
if st.session_state.active_chat_id is None:

    create_new_chat()


current_chat_id = st.session_state.active_chat_id

# Configuration tells LangGraph which conversation to use
CONFIG = {
    "configurable": {
        "thread_id": current_chat_id
    }
}


# -----------------------------
# DISPLAY OLD MESSAGES
# -----------------------------

# Get conversation history from SQLite
state = workflow.get_state(CONFIG)

messages = state.values.get("messages", []) if state.values else []

# Display previous messages
for message in messages:

    if message.type == "human":
        role = "user"

    else:
        role = "assistant"

    with st.chat_message(role):

        st.write(message.content)


# -----------------------------
# USER INPUT
# -----------------------------

user_input = st.chat_input("Ask DotAI anything...")


if user_input:

    # Display user's message
    with st.chat_message("user"):
        st.write(user_input)

    # Display AI response
    with st.chat_message("assistant"):

        response_area = st.empty()

        response_text = ""

        # Stream response from LangGraph
        for message_chunk, metadata in workflow.stream(
            {
                "messages": [HumanMessage(content=user_input)]
            },
            config=CONFIG,
            stream_mode="messages"
        ):

            # Get text from AI response
            if message_chunk.content:

                response_text += message_chunk.content

                response_area.markdown(response_text + "▌")

        # Display completed response
        response_area.markdown(response_text)

    # Refresh to show saved conversation
    st.rerun()