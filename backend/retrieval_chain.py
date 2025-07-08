import logging
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from backend.vector_store import VectorDB
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

logger = logging.getLogger(__name__)

class Retrieval:
    def __init__(self, db):
        self.db = db

    def retrieval_chain(self):
        logger.info("Initializing LLM model and retrieval chain.")
        llm_model = ChatGroq(model="llama-3.3-70b-versatile")

        prompt_template = """
        You are answering as a helpful assistant.

        Role: {role}
        Question: {input}
        Context: {context}

        Provide an answer specifically helpful for the role above.
        If you don't know the answer, just say that you don't know. Don't make anything up.
        """

        prompt = PromptTemplate(
            input_variables=["input", "context", "role"],
            template=prompt_template
        )

        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        logger.info("Retriever and prompt set up.")

        document_chain = create_stuff_documents_chain(llm_model, prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)

        def chain_with_sources(input):
            result = qa_chain.invoke(input)
            sources = [doc.metadata.get("source", "Unknown") for doc in result["context"]]
            return {"answer": result["answer"], "sources": sources}

        logger.info("Retrieval chain ready.")
        return chain_with_sources