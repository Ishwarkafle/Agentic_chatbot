# -----------------------------
# IMPORTS
# -----------------------------

from dotai_fixed_tools import chatbot,get_all_threads


from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

import streamlit as st
import uuid


# -----------------------------
# THREAD FUNCTIONS
# -----------------------------

# Generate a unique ID for every conversation
# CHANGED: the ID now starts with the username
def generate_thread_id():

    return f"{username}__{uuid.uuid4()}"


# Add a thread to the conversation list
def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"].append(thread_id)


# Create a new conversation
def reset_chat():

    st.session_state["thread_id"] = generate_thread_id()

    st.session_state["message_history"] = []

    add_thread(st.session_state["thread_id"])


# Load a previous conversation
def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖"
)

st.title("Agentic Chatbot with LangGraph")


# -----------------------------
# USERNAME
# -----------------------------

# CHANGED: new block, asks for a name before anything else runs
username = st.sidebar.text_input("Your name").strip().lower()

if not username:

    st.info("Enter your name in the sidebar to start.")

    st.stop()


# -----------------------------
# SESSION STATE
# -----------------------------

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()


# CHANGED: only load this user's conversations
if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = [
        t for t in get_all_threads() if t.startswith(f"{username}__")
    ]


# Add the current conversation to the list
add_thread(st.session_state["thread_id"])


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("My Conversations")


# Create a new chat
if st.sidebar.button("New Chat"):

    reset_chat()

    st.rerun()


# Display previous conversations
for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        # Convert LangChain messages into Streamlit format
        for message in messages:

            if isinstance(message, HumanMessage):

                role = "user"

            elif isinstance(message, AIMessage):

                role = "assistant"

            else:

                continue

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_messages

        st.rerun()


# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -----------------------------
# CHAT INPUT
# -----------------------------

user_input = st.chat_input("Type here")


if user_input:

    # Save the user's message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display the user's message
    with st.chat_message("user"):

        st.markdown(user_input)


    # Configuration for the current conversation
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


    # -----------------------------
    # STREAM AI RESPONSE
    # -----------------------------

    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }


        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                config=CONFIG,

                stream_mode="messages"
            ):

                # Show which tool is being executed
                if isinstance(message_chunk, ToolMessage):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`...",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`...",
                            state="running",
                            expanded=True
                        )


                # Stream only AI-generated text
                if isinstance(message_chunk, AIMessageChunk):

                    if isinstance(message_chunk.content, str):

                        yield message_chunk.content


        # Display the streamed response
        ai_message = st.write_stream(
            ai_only_stream()
        )


        # Update tool execution status
        if status_holder["box"] is not None:

            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False
            )


    # Save the assistant's response
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })


    # Refresh the application
    st.rerun()