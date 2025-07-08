from backend.text_chunks import Chunk
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

vector_cache_dir = "vectorstore/cache/"
vector_index_path = os.path.join(vector_cache_dir, "faiss_index")

class VectorDB:
    def __init__(self, chunks):
        self.embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        self.chunks = chunks
        os.makedirs(vector_cache_dir, exist_ok=True)

    def vectorstore_(self):
        if os.path.exists(os.path.join(vector_index_path, "index.faiss")):
            logger.info("Loaded vectorstore from cache.")
            return FAISS.load_local(
                folder_path=vector_index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )

        db = FAISS.from_documents(self.chunks, self.embeddings)
        db.save_local(vector_index_path)
        logger.info("Vectorstore built and cached.")
        return db
