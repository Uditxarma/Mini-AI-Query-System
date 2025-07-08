# Mini AI Query System

## Setup Instructions

1. **Clone the repository** and navigate to the project directory.
2. **Install Python 3.11+** (recommended).
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add your PDF files** to the `docs/` directory.
5. **Start the FastAPI server:**
   ```bash
   uvicorn frontend.api:app --reload
   ```
6. **Open the Web UI:**
   Open [http://localhost:8000/index.html](http://localhost:8000/index.html) in your browser to use the chatbot interface.

7. **(Optional) Access the API docs:**
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

## Libraries Used
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)
- [langchain](https://python.langchain.com/)
- [langchain_ollama](https://github.com/langchain-ai/langchain-ollama)
- [langchain_community](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OllamaEmbeddings]
- [ChatGroq]
- [PyPDF]

## Sample Queries to Test
Send a POST request to `/query` endpoint with JSON body:
```json
{
  "query": "What is the title of the document?",
  "role": "General"
}
```
Sample response:
```json
{
  "answer": "The title of the document is 'The Black Cat'.",
  "sources": ["The Black Cat Author Edgar Allan Poe.pdf"]
}
```

Send feedback to `/feedback` endpoint:
```json
{
  "query": "What is the title of the document?",
  "answer": "The Black Cat",
  "helpful": true,
  "user_comment": "Accurate answer."
}
```

## Notes on Limitations or Assumptions
- Only PDF files in the `docs/` directory are processed.
- The system automatically detects and indexes new or changed PDFs on restart.
- The vectorstore is cached for faster startup; cache is invalidated if PDFs change.
- Only basic role-based context is supported.
- The system is designed for demonstration and may not scale for large document sets or high concurrency.
- Ensure all dependencies are installed and compatible with your Python version.

---
For any issues, please check the logs or open an issue in the repository.
