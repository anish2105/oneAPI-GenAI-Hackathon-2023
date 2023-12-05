import streamlit as st
import os
import pdfplumber
from langchain.document_loaders.base import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings

# Set environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


# Embedding
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embedding-s-en-v1")

# Load PDF documents
def pdf_loader(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        pages = pdf.pages

        documents = []
        for page in pages:
            text = page.extract_text()
            documents.append(Document(page_content=text))

    return documents

documents = pdf_loader("legal_women.pdf")

# Text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(docs, embeddings)

# LLM
repo_id = 'HuggingFaceH4/zephyr-7b-beta'
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"max_length": 128, "temperature": 0.5}
)
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=db.as_retriever(k=2),
                                 return_source_documents=True,
                                 verbose=True,
                                 )

# Streamlit app
st.title("SenOR - AI for Lawyers")
st.subheader("Chat with SenOR")
chat_history = []

# Chat input
user_query = st.text_area("User Input", "")
if st.button("Submit"):
    # Add user query to chat history
    chat_history.append({"role": "user", "message": user_query})

    # Get SenOR's response
    senor_response = qa(user_query)["result"]

    # Add SenOR's response to chat history
    chat_history.append({"role": "senor", "message": senor_response})

for entry in chat_history:
    if entry["role"] == "user":
        st.text(f"User: {entry['message']}")
    else:
        st.text(f"SenOR: {entry['message']}")
