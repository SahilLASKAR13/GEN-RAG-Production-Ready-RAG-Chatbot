from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 RAG Chatbot")
st.write("Upload your documents and ask questions about them.")

# -------------------------------
# Session State
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "json"],
    )

    if uploaded_file:

        if st.button("Upload"):

            with st.spinner("Uploading and indexing..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                )

                print("Status:", response.status_code)
                print(response.text)

            if response.status_code == 200:

                data = response.json()

                st.success(
                    f"""
Indexed Successfully!

Chunks : {data["chunks"]}
"""
                )

            else:

                st.error(response.text)

# -------------------------------
# Chat History
# -------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# -------------------------------
# Chat Input
# -------------------------------

question = st.chat_input(
    "Ask anything about your documents..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    payload = {
        "query": question,
        "chat_history": st.session_state.messages[:-1],
    }

    with st.spinner("Thinking..."):

        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
        )

    if response.status_code == 200:

        data = response.json()

        answer = data["answer"]

        with st.chat_message("assistant"):

            st.markdown(answer)

            if data["sources"]:

                with st.expander("Sources"):

                    for source in data["sources"]:

                        page = source.get("page")

                        if page is None:

                            st.write(
                                f"""
📄 {source['source']}
Score: {source['score']:.3f}
"""
                            )

                        else:

                            st.write(
                                f"""
📄 {source['source']}

Page: {page}

Score: {source['score']:.3f}
"""
                            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    else:

        st.error(response.text)