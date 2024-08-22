from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import os
import uvicorn
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Set your API keys
os.environ["OPENAI_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""

# Initialize FastAPI
app = FastAPI()

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Function to process and index a PDF
def process_and_index_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter()
    text_chunks = text_splitter.split_documents(data)
    
    index_name = "dog"
    docsearch = Pinecone.from_documents(text_chunks, embeddings, index_name=index_name, namespace='poochpw')
    
    return docsearch

# Upload and update index endpoint
@app.post("/update_index/")
async def update_index(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"pdfs/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        
        # Process and update the index with the new PDF
        docsearch = process_and_index_pdf(file_location)
        
        # Initialize the OpenAI model and RetrievalQA with the new index
        llm = OpenAI(temperature=0.9)
        global qa
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
        
        return {"message": "Index updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryList(BaseModel):
    questions: List[str]

@app.post("/query/")
async def query_chatbot(query_list: QueryList):
    try:
        results = []
        for question in query_list.questions:
            try:
                # First, try to get the answer from the PDF-based QA
                answer = qa.run(question)
                if not answer:
                    raise Exception("No answer from PDF-based QA.")
            except Exception as e:
                # If PDF-based QA fails, use OpenAI directly
                llm = OpenAI(temperature=0.9)
                prompt = f"Answer the following question using your general knowledge: {question}"
                answer = llm(prompt).choices[0].text.strip()
                
            results.append({"question": question, "answer": answer})
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
