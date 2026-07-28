# Cadence Innovus AI Assistant: Physical Design Co-Pilot

# Motivation
As an Electronics and Communication Engineering (ECE) student, I quickly realized that the backend process of ASIC development, specifically Physical Design (Place and Route, CTS, Routing)—is incredibly complex, highly iterative, and deeply reliant on proprietary scripting. Engineers spend countless hours debugging timing violations and congestion using tool-specific Tcl commands.

I wanted to bridge the gap between traditional VLSI workflows and modern Artificial Intelligence. I built this AI assistant to specifically target Cadence Innovus, acting as a dedicated co-pilot that generates accurate, heavily-commented Tcl scripts for the entire physical design flow so engineers can focus on architecture rather than syntax.

# Project Overview
This project is a specialized Retrieval-Augmented Generation (RAG) platform. Instead of relying on a general AI that might hallucinate fake commands, this assistant is explicitly trained on a curated database of Cadence Innovus manuals, official command references, and custom Tcl scripts. When a user asks a physical design question, the AI searches its local vector database for the factual syntax, and then leverages Google's Gemini model to generate a deployable, ready-to-use code block.

#Technologies & Tools Used

User Interface: Streamlit (For a clean, ChatGPT-style web experience) 


Core LLM: Google Gemini API (gemini-flash-lite-latest for high-speed, scalable inference) 


AI Framework: LangChain (To orchestrate the RAG pipeline and connect the LLM to the database) 


Vector Database: ChromaDB (For local, persistent storage of document embeddings) 


Embeddings: HuggingFace sentence-transformers (all-MiniLM-L6-v2) 


Environment: Python, Google Colab (for initial data processing), GitHub, Streamlit Community Cloud (for deployment).

#Core Concepts Implemented

Retrieval-Augmented Generation (RAG): Grounding the AI's responses in factual, uploaded documentation to eliminate hallucinations.


Vector Embeddings & Chunking: Processing massive 500-page EDA manuals by splitting them into 1000-character chunks and converting them into mathematical coordinates for instant semantic search.


Mega-Prompt Engineering: Forcing the LLM into a specific persona ("Expert Physical Design Engineer") and strictly enforcing standard Cadence Tcl syntax with detailed # comments for every command.


Cloud Deployment & Secrets Management: Transitioning from a local environment to a permanent cloud server while securely hiding API keys in environment variables.

# Step-by-Step Flow of the Application

Data Ingestion: Cadence Innovus PDFs, .tcl scripts, and .json reference files are loaded into the system.


Processing: The text is chunked and embedded into a ChromaDB vector database, giving the AI a permanent "memory." 

User Query: The user inputs a physical design prompt via the Streamlit web interface (e.g., "Write the commands for a floorplan with 70% utilization").


Retrieval: The LangChain retriever searches the ChromaDB database for the top 5 most relevant command chunks.


Generation: The retrieved syntax and the user's prompt are sent to the Gemini model, which synthesizes a perfectly formatted Tcl script.

Output: The script is streamed back to the user interface for easy copying and pasting into the Cadence terminal.

# Key Achievements & Problem Solving
During development, I successfully navigated several complex engineering hurdles, including:

Overcoming major Python "Dependency Hell" by synchronizing core C++ binaries between numpy, pandas, and torchvision in the cloud environment.

Managing cloud architecture limitations by utilizing Google Drive for persistent database storage while operating inside ephemeral Colab instances.

Bypassing API rate limits and Tier-0 quotas by strategically pivoting to subsidized "Lite" inference models.
