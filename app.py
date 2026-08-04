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
st.set_page_config(page_title="Innovus AI Assistant", page_icon="🤖")
st.title("Cadence Innovus AI Assistant")
st.markdown("Ask me any physical design question, and I will generate the exact Tcl commands based on my training data.")

# 2. Securely Fetch API Keys
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
api_key = os.environ.get("GEMINI_API_KEY")

if not supabase_url or not api_key:
    st.error("Missing Environment Variables. Please check Streamlit Secrets.")
    st.stop()

# Initialize Supabase
supabase: Client = create_client(supabase_url, supabase_key)

# Generate Secure Google OAuth Link
oauth_response = supabase.auth.sign_in_with_oauth({
    "provider": "google",
    "options": {
        "redirect_to": "https://innovus-ai-assistant-dxeld9bwwxefq2ztpms63w.streamlit.app"
    }
})

# 3. Sidebar Authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

with st.sidebar:
    st.title("🔐 User Authentication")
    
    if not st.session_state.logged_in:
        # The Fixed Google SSO Button
        st.link_button("🌐 Continue with Google", oauth_response.url)
        
        st.divider()
        st.markdown("### Or use Email:")
        
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sign In"):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.logged_in = True
                    st.session_state.user_email = res.user.email
                    st.rerun()
                except Exception as e:
                    st.error("Login Failed. Check credentials.")
        with col2:
            if st.button("Sign Up"):
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Account created! You can now log in.")
                except Exception as e:
                    st.error("Signup Failed.")
    else:
        st.success(f"Logged in as: {st.session_state.user_email}")
        if st.button("Log Out"):
            supabase.auth.sign_out()
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.rerun()

# Stop execution if not logged in
if not st.session_state.logged_in:
    st.warning("Please log in from the sidebar to access the AI Co-Pilot.")
    st.stop()

# 4. Load the Database & AI (Cached for speed)
@st.cache_resource
def load_ai_chain():
    db_folder = './database'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_folder, embedding_function=embeddings)
    
    # Using the free-tier champion model with native multimodality
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=api_key)
    
    # Expanded search window (k=10) for deeper context
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    
    # Loosened prompt for expert explanations + strict syntax
    template = """You are an Expert Physical Design Engineer specializing in Cadence Innovus. 
    Use the following retrieved documents as your primary source of truth for syntax to prevent errors. 
    However, use your vast internal engineering knowledge to explain the theory, the 'why' behind the commands, 
    and provide a highly detailed, step-by-step breakdown of the process.
    
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

# 5. Multimodal Inputs (Image & Voice)
with st.expander("📸 Upload Layout Screenshots or 🎙️ Voice Prompts"):
    uploaded_image = st.file_uploader("Upload Congestion/DRC Map (JPG/PNG)", type=['png', 'jpg', 'jpeg'])
    audio_bytes = st.audio_input("Record a voice prompt")

# 6. Streamlit Chat Interface
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
                # Add notes to the prompt if audio/image exists (basic handling)
                if uploaded_image:
                    prompt += "\n[User attached a layout image to this request]"
                if audio_bytes:
                    prompt += "\n[User attached an audio file to this request]"
                
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Save to Supabase Cloud Database
                supabase.table("chat_history").insert({
                    "user_email": st.session_state.user_email,
                    "prompt": prompt,
                    "response": response
                }).execute()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
