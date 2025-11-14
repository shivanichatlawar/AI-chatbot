"""
Main Chatbot Application
Streamlit-based UI for PDF chatbot
"""

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pdf_processor import PDFProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot - Gen AI",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


def initialize_chatbot(api_key: str, pdf_path: str, model_name: str = "gpt-3.5-turbo"):
    """Initialize the chatbot with PDF processing"""
    try:
        with st.spinner("Processing PDF and setting up chatbot..."):
            # Process PDF
            processor = PDFProcessor(pdf_path)
            vector_store = processor.process_pdf(api_key=api_key)
            
            # Initialize LLM
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                openai_api_key=api_key
            )
            
            # Initialize memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create retrieval chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=memory,
                return_source_documents=True,
                verbose=False
            )
            
            st.session_state.vector_store = vector_store
            st.session_state.chain = chain
            st.session_state.pdf_processed = True
            
        return True
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return False


def main():
    st.title("🤖 PDF Chatbot with Gen AI")
    st.markdown("Ask questions about the PDF document using AI-powered chat")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key. You can also set it in .env file as OPENAI_API_KEY"
        )
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0,
            help="Choose the OpenAI model to use"
        )
        
        # PDF path
        pdf_path = st.text_input(
            "PDF File Path",
            value="Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf",
            help="Path to the PDF file"
        )
        
        # Initialize button
        if st.button("🔄 Initialize/Reload Chatbot", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key")
            elif not os.path.exists(pdf_path):
                st.error(f"PDF file not found: {pdf_path}")
            else:
                st.session_state.messages = []
                st.session_state.pdf_processed = False
                if initialize_chatbot(api_key, pdf_path, model_name):
                    st.success("Chatbot initialized successfully!")
        
        # Status
        st.divider()
        st.subheader("Status")
        if st.session_state.pdf_processed:
            st.success("✅ Chatbot Ready")
        else:
            st.warning("⚠️ Chatbot Not Initialized")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if st.session_state.chain:
                st.session_state.chain.memory.clear()
            st.rerun()
    
    # Main chat interface
    if not st.session_state.pdf_processed:
        st.info("👈 Please configure and initialize the chatbot in the sidebar to get started.")
        st.markdown("""
        ### How to use:
        1. Enter your OpenAI API key in the sidebar
        2. Select the model you want to use
        3. Verify the PDF file path
        4. Click "Initialize/Reload Chatbot"
        5. Start asking questions!
        """)
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📄 View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.text(f"Source {i}: {source.page_content[:200]}...")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chain.invoke({"question": prompt})
                        answer = response["answer"]
                        sources = response.get("source_documents", [])
                        
                        st.markdown(answer)
                        
                        # Show sources
                        if sources:
                            with st.expander("📄 View Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.text(f"Source {i}: {source.page_content[:300]}...")
                        
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })


if __name__ == "__main__":
    main()

