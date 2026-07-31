import os
import streamlit as st
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="Innovus AI Full-Stack Platform", page_icon="🤖", layout="wide")
st.title("Cadence Innovus AI Co-Pilot")

# 2. Initialize Supabase Connection using Secrets
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error("Supabase credentials missing in Streamlit Secrets! Please add SUPABASE_URL and SUPABASE_KEY.")
    st.stop()

# Securely fetch Gemini API key
gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not gemini_key:
    st.error("GEMINI_API_KEY missing in Secrets!")
    st.stop()

# 3. Sidebar Authentication (Sign In / Sign Up)
st.sidebar.header("🔐 User Authentication")
auth_mode = st.sidebar.radio("Choose Mode", ["Sign In", "Sign Up"])
user_email = st.sidebar.text_input("Email Address")
user_password = st.sidebar.text_input("Password", type="password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""

if st.sidebar.button("Submit"):
    try:
        if auth_mode == "Sign Up":
            res = supabase.auth.sign_up({"email": user_email, "password": user_password})
            st.sidebar.success("Account created! Please sign in.")
        else:
            res = supabase.auth.sign_in_with_password({"email": user_email, "password": user_password})
            st.session_state.logged_in = True
            st.session_state.user_email = user_email
            st.sidebar.success("Successfully logged in!")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Auth Error: {str(e)}")

if not st.session_state.logged_in:
    st.warning("Please sign in or sign up via the sidebar to access the AI co-pilot.")
    st.stop()

st.sidebar.success(f"Logged in as: {st.session_state.user_email}")
if st.sidebar.button("Sign Out"):
    supabase.auth.sign_out()
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.rerun()

# 4. Load Database & AI Brain (Upgraded with k=10 search window)
@st.cache_resource
def load_ai_chain():
    db_folder = './database'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_folder, embedding_function=embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=gemini_key)
    
    # Widen search window from 5 to 10 for detailed context
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # Loosened expert prompt balancing strict syntax with detailed explanation
    template = """You are an Expert Physical Design Engineer specializing in Cadence Innovus. 
    Use the provided documents as your primary source of truth for exact Tcl syntax to prevent errors. 
    In addition, use your vast internal engineering knowledge to explain the theory, the 'why' behind the commands, 
    and provide a highly detailed, comprehensive, step-by-step breakdown of the process.
    
    Context: {context}
    
    Chat History / Previous Messages: {history}
    
    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough(), "history": lambda x: st.session_state.get("chat_history_summary", "")}
        | prompt
        | llm
        | StrOutputParser()
    )

try:
    rag_chain = load_ai_chain()
except Exception as e:
    st.error(f"Error loading database files. Details: {str(e)}")
    st.stop()

# 5. Chat Interface & Supabase Logging
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if prompt := st.chat_input("How do I create a floorplan with 70% utilization?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Maintain lightweight history string for memory context
    st.session_state["chat_history_summary"] = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])

    with st.chat_message("assistant"):
        with st.spinner("Searching manuals and formulating detailed engineering breakdown..."):
            try:
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Save prompt and response securely to Supabase table
                supabase.table("chat_history").insert({
                    "user_email": st.session_state.user_email,
                    "prompt": prompt,
                    "response": response
                }).execute()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
