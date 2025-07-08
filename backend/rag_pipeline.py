from backend.text_chunks import Chunk
from backend.vector_store import VectorDB
from backend.retrieval_chain import Retrieval
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
import logging
import glob
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Pipeline:
    def pipe(self):
        self.chunks = Chunk().chunk_documents()
        self.stored_vector = "vectorstore/cache/faiss_index"
        self.embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")

        # --- Hash logic for PDF change detection ---
        hash_file = os.path.join(self.stored_vector, "pdf_hash.txt")
        pdf_files = sorted(glob.glob("docs/**/*.pdf", recursive=True))
        hash_md5 = hashlib.md5()
        for pdf_file in pdf_files:
            with open(pdf_file, "rb") as f:
                while chunk := f.read(8192):
                    hash_md5.update(chunk)
        current_hash = hash_md5.hexdigest()
        hash_changed = True
        if os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                hash_changed = False

        faiss_index_path = os.path.join(self.stored_vector, "index.faiss")
        if os.path.exists(faiss_index_path) and not hash_changed:
            logger.info("Loading vectorstore from cache in pipeline...")
            self.db = FAISS.load_local(
                folder_path=self.stored_vector,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("Building new vectorstore in pipeline...")
            self.db = VectorDB(self.chunks).vectorstore_()
            # Save the new hash
            os.makedirs(self.stored_vector, exist_ok=True)
            with open(hash_file, "w") as f:
                f.write(current_hash)
        
        self.chain = Retrieval(self.db).retrieval_chain()
        logger.info("Pipeline ready with retrieval chain.")
        return self.chain
        
    def answer_query(self, query, role="General"):
        logger.info(f"Received query from role '{role}': {query}")
        result = self.chain({"input": query, "context": "", "role": role})
        logger.info("LLM returned response successfully.")
        return result["answer"], result["sources"]
