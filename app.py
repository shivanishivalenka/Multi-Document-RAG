import streamlit as st
import subprocess
import os  
from dotenv import load_dotenv  
from langchain_mistralai import ChatMistralAI  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
import pypdf


load_dotenv()  
mistral_api_key = os.getenv("MISTRAL_API_KEY")  
if not mistral_api_key:  
    raise ValueError("Mistral API key not found. Please set it in .env file.")  
llm = ChatMistralAI(  
model="mistral-large-latest",   
api_key=mistral_api_key  
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
prompt = hub.pull('pwoc517/crafted_prompt_for_rag')
vector_store = InMemoryVectorStore(embeddings)


def retrieve(question):
    return vector_store.max_marginal_relevance_search(question,k=4)

def generate(question,context):
    docs_content = "/n/n".join(ever.page_content for ever in context)
    message = prompt.invoke({"question" : question, "context" : docs_content})
    response = llm.invoke(message)
    return response.content

     

def rag(question):
    print("function rag called")
    docs = loader.load()
    print("splitting data")
    textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 300)
    all_splits = textSplitter.split_documents(docs)
    
    _ = vector_store.add_documents(documents= all_splits)
    print("retrieving context")
    context = retrieve(question)
    
    print("generating answer")
    answer = generate(question, context)

    return answer


st.set_page_config(page_title="Personalized RAG application", layout="wide")
st.title("RAG APPLICATION")



uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")  
if uploaded_file is not None:
    print(type(uploaded_file))
    with open(uploaded_file.name,"wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader(uploaded_file.name)

if "messages" not in st.session_state:
    st.session_state["messages"] = []



st.subheader("Chat Area")
user_input = st.text_input("Your message:", placeholder="Type your message here and press Enter...")



if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    response = rag(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": response})


for message in st.session_state["messages"]:
    role = "**You:**" if message["role"] == "user" else "**LLM:**"
    st.markdown(f"{role} {message['content']}")




