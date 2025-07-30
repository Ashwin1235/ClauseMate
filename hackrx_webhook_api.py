from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_community.chat_models import ChatOllama
import os
import uvicorn

app = FastAPI()

# Initialize Embeddings and Vector Store
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"}
)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory="chroma_db",
)

# Load Ollama Chat Model
llm = ChatOllama(model="mistral")

# Input Schema
class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# Download PDF and save locally
def download_pdf(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise HTTPException(status_code=400, detail="Failed to download document.")

# Process Document and Update Vector DB
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(raw_documents)
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)

# Retrieval + RAG Answer Generation with Enhanced Filtering
def retrieve_and_answer(questions):
    answers = []
    for q in questions:
        docs = vector_store.similarity_search(q, k=5)
        filtered_docs = [doc.page_content for doc in docs if any(keyword.lower() in doc.page_content.lower() for keyword in q.split())]

        context = "\n\n".join(filtered_docs) if filtered_docs else "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a policy clause extraction assistant. Given the context, extract the exact clause(s) or sentences that answer the user's question. Do not summarize or infer. If the answer is not found in the context, reply 'Not Found'.

Context:
{context}

Question: {q}

Answer:
"""
        response = llm.invoke(prompt)
        answers.append(response.content.strip())

    return answers

@app.post("/api/v1/hackrx/run")
def run_submission(req: RunRequest):
    pdf_path = "temp_doc.pdf"
    download_pdf(req.documents, pdf_path)
    process_document(pdf_path)
    answers = retrieve_and_answer(req.questions)
    return {"answers": answers}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
