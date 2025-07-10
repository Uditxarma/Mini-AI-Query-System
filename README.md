# Mini AI Query System

> **Note:** On first run, the system may take some time to start as it generates embeddings and stores them in the vector database. Subsequent runs will be much faster unless you add or change documents.

## Project Overview
This Retrieval-Augmented Generation (RAG) system lets you ask questions about your PDF documents and get AI-generated answers with clear source references. It’s designed for easy, transparent document search using natural language.

## How It Works
- Place your PDF files in the `docs/` folder. The system detects new or changed PDFs on each start.
- On launch, it processes new/updated files, splits them into chunks, and creates vector embeddings using OllamaEmbeddings.
- Embeddings are stored in a FAISS vector database for fast similarity search.
- The FastAPI backend provides two endpoints: `/query` (submit questions) and `/feedback` (user feedback on answers).

## Key Features
- Automatic detection and processing of new/changed PDFs
- Fast, relevant retrieval using FAISS
- Role-based context for tailored answers
- Source references in every answer
- Simple feedback mechanism and modern web UI

## Limitations
- Only PDF files are supported by default (can be extended for DOCX/TXT)
- Embedding and LLM models are swappable
- Designed for demo/small-medium sets; optimize for large-scale use

## Project Structure

```
Mini-AI-Query-System/
├── backend/
│   ├── __init__.py
│   ├── rag_pipeline.py
│   ├── retrieval_chain.py
│   ├── text_chunks.py
│   └── vector_store.py
├── docs/
│   └── <your-pdf-files>.pdf
├── frontend/
│   ├── api.py
│   └── index.html
├── vectorstore/
│   └── cache/
│       └── faiss_index/
├── images/
│   └── sample.png
├── requirements.txt
├── Dockerfile
├── README.md
└── ...
```

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
   Go to [http://localhost:8000/index.html](http://localhost:8000/index.html) in your browser.
7. **(Optional) API docs:**
   Go to [http://localhost:8000/docs](http://localhost:8000/docs)

## Libraries Used
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)
- [langchain](https://python.langchain.com/)
- [langchain_ollama](https://github.com/langchain-ai/langchain-ollama)
- [langchain_community](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- OllamaEmbeddings
- ChatGroq
- PyPDF

## Sample Queries to Test
Send a POST request to `/query` endpoint:
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

## Notes
- Only PDFs in `docs/` are processed.
- Vectorstore is cached for faster startup; cache is invalidated if PDFs change.
- Basic role-based context is supported.
- For issues, check logs or open a GitHub issue.

---

## 📹 Video Walkthrough

<p align="center">
  <a href="sample/Live.mp4" target="_blank">
    <img src="sample/Sample.png" alt="Watch the video walkthrough" width="600" style="border-radius:12px; box-shadow:0 2px 8px #0002;"/>
  </a>
</p>

<p align="center">
  <b>Click the image above to watch a short video demo of the Mini AI Query System.</b>
</p>


