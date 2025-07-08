from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_pipeline import Pipeline
from fastapi.staticfiles import StaticFiles
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__)), html=True), name="static")
pipeline = Pipeline()
pipeline.pipe()

class QueryInput(BaseModel):
    query: str
    role: str = "General"

class FeedbackInput(BaseModel):
    query: str
    answer: str
    helpful: bool
    user_comment: str = ""

@app.post("/query")
async def query_rag(input: QueryInput):
    logger.info(f"Received API query from role '{input.role}': {input.query}")
    try:
        answer, sources = pipeline.answer_query(input.query, role=input.role)
        logger.info("Answer generated successfully.")
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return {"error": str(e)}


@app.post("/feedback")
async def feedback(input: FeedbackInput):
    logger.info("Feedback received. Logging to file...")
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "feedback_log.txt")
    with open(log_path, "a") as f:
        f.write(f"{input.query} | {input.answer} | {input.helpful} | {input.user_comment}\n")
    logger.info("Feedback logged successfully.")
    return {"status": "feedback logged"}