import os
import pickle
import glob
import hashlib
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

file_path = "docs/"
cache_dir = "vectorstore/"
chunk_cache_file = os.path.join(cache_dir, "chunk_cache.pkl")
hash_cache_file = os.path.join(cache_dir, "pdf_hash.txt")

class Chunk:
    def __init__(self, docs_path=file_path):
        self.docs_path = docs_path

    def _get_pdf_hash(self):
        hash_md5 = hashlib.md5()
        for pdf_file in sorted(glob.glob(f"{self.docs_path}/**/*.pdf", recursive=True)):
            with open(pdf_file, "rb") as f:
                while chunk := f.read(8192):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load_documents(self):
        loader = DirectoryLoader(self.docs_path, glob="**/*.pdf", show_progress=True)
        return loader.load()

    def chunk_documents(self):
        os.makedirs(cache_dir, exist_ok=True)
        current_hash = self._get_pdf_hash()

        if os.path.exists(chunk_cache_file) and os.path.exists(hash_cache_file):
            with open(hash_cache_file, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                logger.info("Loaded chunks from cache (PDFs unchanged).")
                with open(chunk_cache_file, "rb") as f:
                    return pickle.load(f)

        logger.info("PDFs changed or cache missing — regenerating chunks...")
        documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        with open(chunk_cache_file, "wb") as f:
            pickle.dump(chunks, f)
        with open(hash_cache_file, "w") as f:
            f.write(current_hash)

        logger.info("Chunks processed and saved to cache.")
        return chunks

if __name__ == "__main__":
    chunks = Chunk().chunk_documents()
    logger.info(f"Total chunks: {len(chunks)}")
