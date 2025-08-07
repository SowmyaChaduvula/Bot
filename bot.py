
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set or loaded. Please check your .env file.")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

import streamlit as st
st.set_page_config(
    page_title="Agent DevNext",
    page_icon=":material/robot:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import warnings
warnings.filterwarnings("ignore")
__import__('pysqlite3')

import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

data_directory = os.path.join(os.path.dirname(__file__), "data")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

hf_hub_llm = Ollama(model = 'llama3')

prompt_template="Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),  # retriever is set to fetch top 3 results
    chain_type_kwargs={"prompt": custom_prompt})

def get_response(question):
    print("into response................................................")
    result = rag_chain.run(question)
    print(result,"resssssssssssssssssssssssss")
    response_text = result
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app
# Remove whitespace from the top of the page and sidebar
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )
st.title("Welcome to the Bot Interface🤖")
st.subheader("Agent DevNext", divider="gray", anchor=False)

initial_message = """
    Hi there! I'm a devNext Agent 🤖\n
    How can I help You..!
    """
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching...."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = response  # Directly use the response
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
