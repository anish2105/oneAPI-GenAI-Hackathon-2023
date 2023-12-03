#imports
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
from langchain.llms import HuggingFaceHub
import pdfplumber
from langchain.document_loaders.base import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
#Embedding
from langchain.embeddings import OpenAIEmbeddings
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()
#RAG
def pdf_loader(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        pages = pdf.pages

        documents = []
        for page in pages:
            text = page.extract_text()
            documents.append(Document(page_content=text))

    return documents

documents = pdf_loader("legal_women.pdf")

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200
    )

docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(docs, embeddings)



#prompt


#llm
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


query = "what does article 14 says"
result = qa(query)
print(result['result'])