from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import  PineconeVectorStore
import os
from dotenv import load_dotenv
from uuid import uuid4
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if TOGETHER_API_KEY:
    os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Load pdf file


def load_pdf_file(folder_name):
    loader = PyPDFDirectoryLoader(folder_name, glob="*.pdf")
    docs = loader.load()
    return docs



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings

# def language_detection(text):
# #     prompt = f"""Identify the language of the following text. Respond with only the name of the language.

# # Text: "{text}"

# # Answer:"""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "you are an expert to detect the language of the {text}"),
#         ("user", "Text: {text}"),
#         ("user", "Answer:")
#     ])
#     llm = ChatGroq(model="llama3-8b-8192", temperature=0)
#     chain = prompt | llm | StrOutputParser()
#     response = chain.invoke({"text": text})
#     return response

def language_detection(text):
    prompt = ChatPromptTemplate.from_messages([
        (
            "user",
            "Identify the language of the following text. Respond with only the name of the language.\n\nText: {text}\n\nAnswer:"
        )
    ])

    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)  # deterministic
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"text": text})
    return response.strip()

def translate(text, target_language="English"):
    prompt = ChatPromptTemplate.from_messages([
        (
            "user",
            "Translate the following text to {target_language}. Respond with only the translated text.\n\nText: {text}\n\nTranslation:"
        )
    ])

    llm = ChatGroq(model="llama3-8b-8192", temperature=0.0)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"text": text, "target_language": target_language})
    return response.strip()






