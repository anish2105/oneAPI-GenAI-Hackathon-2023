import streamlit as st
import os
import pdfplumber
from langchain.document_loaders.base import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import HuggingFaceHub
from langchain.embeddings import OpenAIEmbeddings


# Set environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ["OPENAI_API_KEY"] = "Openai"

# Embedding
embeddings = OpenAIEmbeddings()

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

#prompt
# prompt = '''
# Your name is Senor, you are an AI Assistant for Lawyers
# Use the following pieces of context to answer the users question, answer should be short
# If you can't find answers in the context, just say "I'm sorry, I only answer for women and childer rights", don't try to make up an answer.
# ALWAYS answer from the perspective of being Senor

# '''


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
# qa.combine_documents_chain.llm_chain.prompt.template = prompt
# Streamlit app
st.title("SenOR - AI for Lawyers")

# Query input
query = st.text_input("Enter your legal query:")
if st.button("Submit"):
    result = qa(query)
    st.write("Answer:", result['result'])

