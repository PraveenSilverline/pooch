from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn
from typing import List
import firebase_admin
from firebase_admin import credentials, firestore
import requests

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from fastapi.middleware.cors import CORSMiddleware

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"C:\Users\athth\OneDrive\Documents\GitHub\Pooch-Paw-ML\Final\Function03\chatbot\poochpaw-test-firebase-adminsdk-72esy-95d92a0b3f.json")

firebase_admin.initialize_app(cred)
db = firestore.client()

# List of origins that should be allowed to make requests
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://192.168.8.100:8000",
    "http://127.0.0.1:8000",
    # Add more origins as needed
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your API keys
os.environ["OPENAI_API_KEY"] = "sk-proj-9lD90nMeWxxo78vnBWMTT3BlbkFJaz6uTg5xRY0GDxOjQA37"
os.environ["PINECONE_API_KEY"] = "dd50eadc-8797-47d7-9c93-1a6b94f5c251"

def download_pdfs_from_firestore():
    docs = db.collection("pet-reports").stream()

    if not os.path.exists("pdfs"):
        os.makedirs("pdfs")

    for doc in docs:
        data = doc.to_dict()
        pet_id = data.get("dog_id")
        pdf_url = data.get("pdfUrl")

        if pet_id and pdf_url:
            pdf_filename = f"pdfs/{pet_id}.pdf"

            response = requests.get(pdf_url)
            if response.status_code == 200:
                with open(pdf_filename, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download PDF for pet ID {pet_id}")
        else:
            print(f"Missing 'uid' or 'pdfUrl' in document ID: {doc.id}")


# Download PDFs from Firestore
download_pdfs_from_firestore()


# Load documents and initialize text splitter
loader = PyPDFDirectoryLoader("pdfs")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
text_chunks = text_splitter.split_documents(data)

# Initialize embeddings and Pinecone vector store
embeddings = OpenAIEmbeddings()
index_name = "dog"
docsearch = Pinecone.from_documents(text_chunks, embeddings, index_name=index_name, namespace='poochpw')

# Initialize the OpenAI model and RetrievalQA
llm = OpenAI(temperature=0.9)
retriever = docsearch.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Define the prompt template
template = '''
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]. Always look first in Vector Store
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat 2 times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    '''

prompt = PromptTemplate.from_template(template)

# Wrap the RetrievalQA chain in a Tool class
search_tool = Tool(
    name="Document Search",
    func=rag_chain.run,
    description="Search for information in the document based on the query."
)

# Create the agent using the LLM and the prompt template
tools = [search_tool]
agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)
agent_executor = AgentExecutor(tools=tools, agent=agent, handle_parsing_errors=True, verbose=True)

class QueryList(BaseModel):
    questions: List[str]
    dog_id: str

@app.post("/query/")

async def query_chatbot(query_list: QueryList):
    try:
        results = []
        for question in query_list.questions:
            # Add dog ID to each question
            formatted_question = f"Dog ID: {query_list.dog_id}. {question}"
            response = agent_executor.invoke({"input": formatted_question})
            answer = response['output']  # Replace 'output' with the correct key
            results.append({"question": question, "answer": answer})
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
