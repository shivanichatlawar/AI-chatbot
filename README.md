# PDF Chatbot with Gen AI

A powerful chatbot application that allows you to ask questions about PDF documents using Generative AI. This application uses LangChain, OpenAI, and ChromaDB to create an intelligent document Q&A system.

## Features

- 📄 **PDF Processing**: Automatically extracts and processes text from PDF files
- 🤖 **AI-Powered Chat**: Uses OpenAI's GPT models to answer questions
- 🔍 **Semantic Search**: Vector-based retrieval for accurate context finding
- 💾 **Persistent Storage**: Saves processed embeddings for faster subsequent loads
- 🎨 **Modern UI**: Clean and intuitive Streamlit interface
- 📚 **Source Citation**: Shows which parts of the document were used to answer questions

## Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**:
   - Option 1: Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Option 2: Enter it directly in the application sidebar

## Usage

1. **Run the application**:
   ```bash
   streamlit run chatbot.py
   ```

2. **Configure the chatbot**:
   - Enter your OpenAI API key in the sidebar
   - Select the model (gpt-3.5-turbo, gpt-4, etc.)
   - Verify the PDF file path (default: `Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf`)
   - Click "Initialize/Reload Chatbot"

3. **Start chatting**:
   - Type your questions in the chat input
   - The chatbot will search the PDF and provide answers
   - View source citations to see which parts of the document were used

## How It Works

1. **PDF Processing**: The PDF is read and text is extracted page by page
2. **Text Chunking**: Text is split into manageable chunks with overlap
3. **Embedding**: Each chunk is converted to a vector embedding using OpenAI
4. **Vector Store**: Embeddings are stored in ChromaDB for fast similarity search
5. **Retrieval**: When you ask a question, relevant chunks are retrieved
6. **Generation**: The LLM generates an answer based on the retrieved context

## Project Structure

```
shivani-law-chatbot/
├── chatbot.py              # Main Streamlit application
├── pdf_processor.py        # PDF processing and vector store creation
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment file
├── README.md              # This file
├── chroma_db/             # Vector database (created automatically)
└── Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf  # Your PDF file
```

## Configuration

### Models
- **gpt-3.5-turbo**: Fast and cost-effective (recommended for most use cases)
- **gpt-4**: More accurate but slower and more expensive
- **gpt-4-turbo-preview**: Latest GPT-4 with improved performance

### Customization
You can modify the following in `pdf_processor.py`:
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `search_kwargs["k"]`: Number of chunks to retrieve (default: 3)

## Troubleshooting

### "OpenAI API key not found"
- Make sure you've set the API key in `.env` or entered it in the sidebar

### "PDF file not found"
- Verify the PDF file path is correct
- Make sure the PDF file is in the project directory

### "Error loading vector store"
- Delete the `chroma_db` folder and reinitialize
- Check if the PDF file has changed

## License

This project is open source and available for personal and commercial use.

## Support

For issues or questions, please check:
- OpenAI API documentation: https://platform.openai.com/docs
- LangChain documentation: https://python.langchain.com/
- Streamlit documentation: https://docs.streamlit.io/

