import streamlit as st
import logging
from backend.rag_pipeline import Pipeline

logging.basicConfig(filename="frontend_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

st.title("Mini AI Query System")

@st.cache_resource
def load_pipeline():
    flow = Pipeline()
    flow.pipe()
    return flow

flow = load_pipeline()

role = st.selectbox("Select your role", ["General", "Manager", "Developer", "Analyst"])
user_query = st.text_input("Ask your query")
submit = st.button("Submit")

if submit and user_query:
    logging.info(f"User submitted query as role '{role}': {user_query}")
    answer, sources = flow.answer_query(user_query, role=role)
    st.markdown(f"**Answer:** {answer}")

    if sources:
        st.markdown("**Source Documents:**")
        for src in sources:
            st.markdown(f"- {src}")

    helpful = st.radio("Was this helpful?", ["Yes", "No"])
    comment = st.text_area("Any comments?")
    if st.button("Submit Feedback"):
        logging.info("Feedback submitted through UI.")
        with open("feedback_log.txt", "a") as f:
            f.write(f"{user_query} | {answer} | {helpful} | {comment}\n")
        st.success("Thanks for your feedback!")