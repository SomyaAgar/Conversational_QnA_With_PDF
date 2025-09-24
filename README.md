# Conversational RAG with PDF Uploads and Chat History
This Streamlit app is a Conversational Retrieval-Augmented Generation (RAG) application that allows users to:
- Upload PDF documents.
- Ask questions about the content of uploaded PDFs.
- Maintain chat history for context-aware answers.

The app leverages **LangChain, FAISS, Groq LLM, and Google Generative AI Embeddings** to provide accurate, concise, and context-aware responses.

### Features
- PDF Uploads – Upload multiple PDFs to create a searchable knowledge base.
- Contextual Retrieval – Questions are answered using the uploaded content.
- Chat History Awareness – Answers take into account the conversation history.
- Concise QA Responses – Generates answers in three sentences max.
- Type of Application: This is a RAG (Retrieval-Augmented Generation) application that combines retrieval from PDF content with LLM-generated answers.

### Tech Stack
- Streamlit – Interactive web UI
- LangChain – LLM orchestration, chains, and agents
- FAISS – Vector database for document embeddings
- Google Generative AI Embeddings – Embedding model for vector search
- Groq LLM – LLM used for response generation

### Installation
1. Clone the repository:
git clone https://github.com/your-username/conversational-rag.git
cd conversational-rag

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies:
pip install -r requirements.txt

4. Environment Setup
The app requires the following API keys:
- Groq API Key – for LLM inference.
- Google Gemini API Key – for embeddings generation.

5. Store these keys in a .env file in the project root:
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_google_gemini_api_key

### Usage
1. Run the app with:
streamlit run app.py
2. Enter your Groq API Key.
3. Upload one or more PDF files.
4. Ask questions about the uploaded documents.

The app will return concise answers based on the documents and the chat history.

### Example Questions
1. "Summarize the methodology section from the uploaded PDFs."
2. "What are the main findings in the research papers I uploaded?"
3. "Explain the conclusion of the last PDF document."

### UI Overview:
<img width="1355" height="672" alt="image" src="https://github.com/user-attachments/assets/fb0559de-bbff-4353-a0fa-fd9f702361e3" />

<img width="1349" height="684" alt="image" src="https://github.com/user-attachments/assets/15fa9375-d08e-45ba-9a0f-8691bca5968e" />


