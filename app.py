import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="Innovus AI Assistant", page_icon="🤖")
st.title("Cadence Innovus AI Assistant")
st.markdown("Ask me any physical design question, and I will generate the exact Tcl commands based on my training data.")

# 2. Securely fetch the API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
st.error("API Key not found. Please set the GEMINI_API_KEY environment variable.")
st.stop()

# 3. Load the Database & AI (Cached for speed)
@st.cache_resource
def load_ai_chain():
db_folder = './database'
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=db_folder, embedding_function=embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=api_key)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

template = """You are an Expert Physical Design Engineer specializing in Cadence Innovus.
Use the following retrieved documents as your primary source of truth.
If the documents lack context, use your expert knowledge to explain best practices.
Context: {context}
Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
return "\n\n".join(doc.page_content for doc in docs)

return (
{"context": retriever | format_docs, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

rag_chain = load_ai_chain()

# 4. Streamlit Chat Interface
if "messages" not in st.session_state:
st.session_state.messages = []

# Display previous chats
for message in st.session_state.messages:
with st.chat_message(message["role"]):
st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How do I create a floorplan with 70% utilization?"):
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
st.markdown(prompt)

# Generate AI response
with st.chat_message("assistant"):
with st.spinner("Searching manuals and thinking..."):
try:
response = rag_chain.invoke(prompt)
st.markdown(response)
st.session_state.messages.append({"role": "assistant", "content": response})
except Exception as e:
st.error(f"Error: {str(e)}")
