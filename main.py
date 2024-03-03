import os.path

import openai
import streamlit as st
from llama_index.core import (ServiceContext, SimpleDirectoryReader,
                              StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.llms.openai import OpenAI

PERSIST_DIR = "./storage"
openai.api_key = st.secrets.openai_key

st.set_page_config(page_title="Chat with instabug chat bot", page_icon="./instabug.png", layout="centered", initial_sidebar_state="auto", menu_items=None)


st.title("Chat with instabug docs 💬")

st.info("Powered by [Instabug](https://www.instabug.com/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about instabug"}
    ]

def loadData():
  with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
    if not os.path.exists(PERSIST_DIR):  # not found
      print("Not Found, Creating the index...")
      service_context = ServiceContext.from_defaults(
          llm=OpenAI(model="gpt-4", temperature=0.1))
      
      documents = SimpleDirectoryReader("./documents/").load_data()

      index = VectorStoreIndex.from_documents(
          documents, service_context=service_context, show_progress=True)
      index.storage_context.persist(persist_dir=PERSIST_DIR)

    else:  # found
      print("Found, Loading the index...")
      storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
      index = load_index_from_storage(storage_context)
  
    return index


if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        index = loadData()
        st.session_state.chat_engine =  index.as_query_engine(similarity_top_k=4, response_mode="compact")

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history