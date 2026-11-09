# Import libraries

import streamlit as st
from langchain_core.messages import HumanMessage
from DotAi import workflow
import uuid


# --------------------------------
# 1. PAGE TITLE
# --------------------------------

st.title("DotAi Chatbot")


# --------------------------------
# 2. CREATE SESSION STATE
# --------------------------------

# Store all conversations

if 'chats' not in st.session_state:

    st.session_state.chats = {}


# Store the currently selected conversation

if 'active_chat_id' not in st.session_state:

    st.session_state.active_chat_id = None


# --------------------------------
# 3. CREATE NEW CHAT FUNCTION
# --------------------------------

def create_new_chat():

    # Generate a unique ID for every new chat

    new_chat_id = str(uuid.uuid4())

    # Create a new conversation

    st.session_state.chats[new_chat_id] = {

        'title': 'New Chat',

        'history': []
    }

    # Make this conversation active

    st.session_state.active_chat_id = new_chat_id


# --------------------------------
# 4. CREATE FIRST CHAT
# --------------------------------

# If no conversation exists, create one

if st.session_state.active_chat_id is None:

    create_new_chat()


# --------------------------------
# 5. SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("💬 Chat History")

    # Button to create a new conversation

    if st.button("+ New Chat", use_container_width=True):

        create_new_chat()

        st.rerun()


    st.divider()

    st.write("Previous Conversations")


    # Display all conversations

    # Display conversations from newest to oldest

    for chat_id, chat_data in reversed(list(st.session_state.chats.items())):

        if st.button(
            chat_data['title'],
            key=chat_id,
            use_container_width=True
        ):

            # Switch to selected conversation

            st.session_state.active_chat_id = chat_id

            st.rerun()


# --------------------------------
# 6. GET CURRENT CHAT
# --------------------------------

# Get the selected conversation ID

current_chat_id = st.session_state.active_chat_id

# Get the selected conversation data

current_chat = st.session_state.chats[current_chat_id]


# --------------------------------
# 7. DISPLAY OLD MESSAGES
# --------------------------------

# Display previous messages of this conversation

for message in current_chat['history']:

    with st.chat_message(message['role']):

        st.write(message['content'])


# --------------------------------
# 8. CHAT INPUT
# --------------------------------

user_input = st.chat_input("Type your message...")


if user_input:

    # --------------------------------
    # 9. SAVE USER MESSAGE
    # --------------------------------

    current_chat['history'].append({

        'role': 'user',

        'content': user_input
    })


    # Use the first message as the chat title

    if current_chat['title'] == 'New Chat':

        current_chat['title'] = user_input[:30]


    # Display user message

    with st.chat_message('user'):

        st.write(user_input)


    # --------------------------------
    # 10. CONFIGURE THREAD ID
    # --------------------------------

    # Every chat has its own thread ID

    CONFIG = {

        'configurable': {

            'thread_id': current_chat_id
        }
    }


    # --------------------------------
    # 11. GENERATE STREAMING RESPONSE
    # --------------------------------

    with st.chat_message('assistant'):

        # Get AI response chunk by chunk

        ai_message = st.write_stream(

            message_chunk.content

            for message_chunk, metadata in workflow.stream(

                # Send user message to LangGraph

                {
                    'messages': [
                        HumanMessage(content=user_input)
                    ]
                },

                # Tell LangGraph which conversation to use

                config=CONFIG,

                # Enable message streaming

                stream_mode='messages'
            )

            # Ignore empty or non-text chunks

            if isinstance(message_chunk.content, str)
            and message_chunk.content
        )


    # --------------------------------
    # 12. SAVE AI RESPONSE
    # --------------------------------

    current_chat['history'].append({

        'role': 'assistant',

        'content': ai_message
    })