import streamlit as st
import os
import time
from streamlit import session_state
import pdfplumber
# from langchain.document_loaders.base import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import HuggingFaceHub
# from streamlit_chat import message
from langchain.embeddings import HuggingFaceEmbeddings

# Set environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'your-key'

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embedding-s-en-v1")


# from langchain.document_loaders.csv_loader import CSVLoader
import os
# file_path = os.path.abspath('./rights.csv')
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("legal_women.pdf")
pages = loader.load_and_split()
# Text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(pages)
db = FAISS.from_documents(docs, embeddings)
from langchain import hub
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
def model(user_query,max_length,temp):
    repo_id = 'HuggingFaceH4/zephyr-7b-beta'
    llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"max_length": max_length, "temperature": temp}       )
    qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=db.as_retriever(k=2),
                                 return_source_documents=True,
                                 verbose=True,
                                 chain_type_kwargs={"prompt": prompt})
    return  qa(user_query)["result"]


st.title("🤖 SenOR ")

# Initialize chat history if not present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history with custom styling for larger text
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(f"**🙋🏻‍♂User:** {entry['message']}")
    else:
        st.markdown(f"**🤖SenOR:** {entry['message']}", unsafe_allow_html=True)

# Chatbot interface

with st.sidebar:
    
    st.markdown("<h1 style='text-align:center;font-family:Georgia;font-size:26px;'>🧑‍⚖️ SenOR Legal Advisor </h1>", unsafe_allow_html=True)
    st.markdown("<h7 style='text-align:left;font-size:20px;'>This app is a smart legal chatbot that is integrated into an easy-to-use platform. This would give lawyers "
            "instant access to legal information of Women’s Legal Rights and remove the need for laborious manual research in books or regulations using the power "
            "of Large Language Models</h7>", unsafe_allow_html=True)
    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Features</h1>", unsafe_allow_html=True)
    
    st.markdown(" - Users can adjust token length to control the length of generated responses, allowing for customization based on specific requirements or constraints.")
    st.markdown(" - Users can adjust the temp to control response randomness. Higher values (e.g., 0.5) produce diverse but less focused responses, while low values (e.g., 0.1) result in more focused but less varied answers.")
    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>", unsafe_allow_html=True)
    max_length=st.slider("Token Max Length", min_value=128, max_value=1024, value=128, step=128)
    temp=st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.1)

# Chat input
user_query = st.text_area("User Input", "")
if st.button("Submit"):
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "message": user_query})

    # Get SenOR's response
    senor_response = model(user_query,max_length,temp)

    # Add SenOR's response to chat history
    st.session_state.chat_history.append({"role": "senor", "message": senor_response})

    # Display SenOR's response after submission
    st.markdown(f"**SenOR:** {senor_response}", unsafe_allow_html=True)




























# Chatbot interface
# st.subheader("Chat with SenOR")
# #user_query = st.text_area("User Input", value=st.session_state.user_query, key='user_input')
# # if st.button("Submit"):
# #     # Get SenOR's response
# #     senor_response =model(max_length,temp)


# with st.chat_message("assistant"):
#     message_placeholder = st.empty()
#     full_response = ""
#     # ans = pipeline.run(query = prompt)
#     assistant_response = model(prompt,max_length,temp)


# for chunk in assistant_response.split():
#     full_response += chunk + " "
#     time.sleep(0.05)

#     # Add a blinking cursor to simulate typing
#     message_placeholder.markdown(full_response + "▌")
#     message_placeholder.markdown(full_response)
    
    
    # Add assistant response to chat history
    # st.session_state.messages.append({"role": "senOR", "content": full_response})





    # Add user query and SenOR's response to messages
    # st.session_state.messages.append({"role": "user", "message": user_query})
    # st.markdown(f"**SenOR:** {senor_response}", unsafe_allow_html=True)
    # # Clear user query after submission
    # st.session_state.user_query = ""


















































# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# # Display chat history with custom styling for larger text
# for entry in st.session_state.chat_history:
#     if entry["role"] == "user":
#         st.markdown(f"**User:** {entry['message']}")
#     else:
#         st.markdown(f"**SenOR:** {entry['message']}", unsafe_allow_html=True)
# # Chatbot interface
# st.subheader("Chat with SenOR")

# # Chat input
# user_query = st.text_area("User Input", "")
# if st.button("Submit"):
#     # Add user query to chat history
#     st.session_state.chat_history.append({"role": "user", "message": user_query})

#     # Get SenOR's response
#     senor_response = qa(user_query)["result"]

#     # Add SenOR's response to chat history
#     st.session_state.chat_history.append({"role": "senor", "message": senor_response})

#     # Display SenOR's response after submission
#     st.markdown(f"**SenOR:** {senor_response}", unsafe_allow_html=True)
