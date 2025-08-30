import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os 
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
from dotenv import load_dotenv 
load_dotenv()

# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
api_key = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')

# llm= ChatGroq(groq_api_key = groq_api_key, model= 'gemma2-9b-it')
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
st.markdown(
    "<h1 style='text-align: center; color: green;'>Conversational RAG with PDF uploads and chat history</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'>Upload PDFs and chat with their content.", 
    unsafe_allow_html=True
)
# st.write("Upload PDF's and chat with their content.")

# api_key = st.text_input("Enter your Groq API key:", type = "password")
if api_key:
    llm= ChatGroq(groq_api_key = api_key, model= 'gemma2-9b-it')

    # chat interface
    session_id = st.text_input("Session ID", value = "default_session")

    # statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store ={}

    uploaded_files = st.file_uploader("Choose a PDF file :", type ="pdf", accept_multiple_files = True)
    # process uploaded files:
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf = f"./temp_{uploaded_file.name}"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.read())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)   
            docs=loader.load()
            documents.extend(docs)
        
        # split and create embeddings for the documents 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap =200)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents= splits, embedding= embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt =(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever =create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # answer question prompt 
        system_prompt =(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieval context to answer "
            "the question, If you don't know the answer, say that you  "
            "don't know. Use three sentences maximum and keep the"
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]= ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key = "input",
            history_messages_key = "chat_history",
            output_messages_key ="answer"
        )


        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable":{"session_id": session_id}
                }, 
            )
            # st.write(st.session_state.store)
            st.write("Answer:", response['answer'])
            st.write("Chat History:", session_history.messages)
    
else:
    st.warning("Please enter the Groq API Key")

        








# def create_vector_embedding():
#     # Path to persist FAISS index
#     index_path = "faiss_index"

#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

#         # If FAISS index already exists, load it (no new API calls)
#         if os.path.exists(index_path):
#             st.session_state.vectors = FAISS.load_local(index_path, st.session_state.embeddings, allow_dangerous_deserialization=True)
#             st.write("Loaded existing FAISS index (no new API calls).")

#         else:
#             # Load only 2 documents for now
#             st.session_state.loader = PyPDFDirectoryLoader("researchPapers")
#             st.session_state.docs = st.session_state.loader.load()

#             # Split into chunks
#             st.session_state.text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200
#             )
#             st.session_state.final_documents = st.session_state.text_splitter.split_documents(
#                 st.session_state.docs[:2]
#             )

#             # Create FAISS index (this calls embeddings API ONCE)
#             st.session_state.vectors = FAISS.from_documents(
#                 st.session_state.final_documents,
#                 st.session_state.embeddings
#             )

#             # Save FAISS index for reuse
#             st.session_state.vectors.save_local(index_path)
#             st.write("Created and saved new FAISS index.")
