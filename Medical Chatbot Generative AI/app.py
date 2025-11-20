from flask import Flask, render_template, jsonify, request
from langchain_openai import AzureChatOpenAI
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Fetch API keys safely
PINECONE_API_KEY=
AZURE_OPENAI_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_VERSION= # Updated to a valid API version

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["AZURE_OPENAI_KEY"] = AZURE_OPENAI_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

index_name = "test"

# Load Pinecone index
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_version=AZURE_OPENAI_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    temperature=0.4,
    max_tokens=500
)

# Updated medical check using Runnable interface
medical_check_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a medical query classifier. 
    Determine if the user's question is related to medical topics, health, diseases, treatments, 
    medications, health conditions, wellness, or medical procedures.
    Respond with ONLY 'yes' if it's medical or 'no' if it's not medical."""),
    ("human", "{input}")
])

medical_check_chain = medical_check_prompt | llm

# Main RAG chain for medical answers
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        user_message = request.form['msg']
        print("Received message:", user_message)

        # First check if the question is medical
        medical_check = medical_check_chain.invoke({"input": user_message})
        print("Medical check result:", medical_check)

        # Get just the yes/no answer, trim whitespace and lowercase
        is_medical = medical_check.content.strip().lower() if hasattr(medical_check, 'content') else str(medical_check).strip().lower()

        # Only process medical questions with RAG
        if is_medical == "yes":
            response = rag_chain.invoke({"input": user_message})
            bot_reply = response.get("answer", "No answer found.")
        else:
            bot_reply = "I'm a medical assistant and can only answer medical-related questions. Could you please ask a health-related question instead?"

        return jsonify({'response': bot_reply})
    except Exception as e:
        print("Error in /get endpoint:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)