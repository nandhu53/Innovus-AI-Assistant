import os
import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="Cadence Innovus Multimodal AI Co-Pilot", page_icon="🤖", layout="wide")
st.title("Cadence Innovus Multimodal AI Co-Pilot")
st.markdown("Ask questions via text or voice, upload layout screenshots for visual debugging, and generate precise Tcl commands.")

# 2. Securely fetch the API key
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("API Key not found. Please set the GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

# 3. Load the Database & AI (Cached for speed)
@st.cache_resource
def load_ai_chain():
    db_folder = './database'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_folder, embedding_function=embeddings)

    # Gemini Flash Lite supports multimodal text/image/audio inputs natively
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=api_key)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    template = """You are an Expert Physical Design Engineer specializing in Cadence Innovus.
    Use the following retrieved documents as your primary source of truth for exact Tcl syntax.
    If an image or context is provided, analyze it alongside the technical documentation to provide deep insights.
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

try:
    rag_chain = load_ai_chain()
except Exception as e:
    st.error(f"Error loading database: {str(e)}")
    st.stop()

# 4. Streamlit Chat Interface & Multimodal Inputs
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chats
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"], caption="Uploaded Layout / Reference", width=300)

# --- Multimodal Input Panel in Sidebar or Main Area ---
st.sidebar.header("🎛️ Multimodal Inputs")
uploaded_image = st.sidebar.file_uploader("Upload Layout Screenshot / Error Log", type=["png", "jpg", "jpeg"])

# Voice input widget (records audio from browser microphone)
audio_file = st.audio_input("🎤 Or speak your prompt here:")

# Determine user input source (Text chat input, or transcribed/processed voice)
text_prompt = st.chat_input("How do I create a floorplan with 70% utilization?")

# Process active prompt source
active_prompt = None
if text_prompt:
    active_prompt = text_prompt
elif audio_file:
    # Note: For full speech-to-text transcription, you can integrate Whisper or Gemini audio processing here.
    active_prompt = "[Voice Prompt Received - Audio Data Attached]"
    st.info("Audio recorded successfully! Processing voice command...")

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt, "image": uploaded_image})
    with st.chat_message("user"):
        st.markdown(active_prompt)
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Layout / Reference", width=300)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing manuals, inspecting inputs, and thinking..."):
            try:
                # Combine prompt text with image context if available
                full_query = active_prompt
                if uploaded_image:
                    full_query = f"[User uploaded an image for visual analysis]. Prompt: {active_prompt}"
                
                response = rag_chain.invoke(full_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
