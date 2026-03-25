import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("Lab 6")

if "anthropic_api_key" not in st.secrets:
    st.error("Add `anthropic_api_key` to `.streamlit/secrets.toml` for Claude.")
    st.stop()

llm = init_chat_model(
    "claude-haiku-4-5-20251001",
    model_provider="anthropic",
    api_key=st.secrets["anthropic_api_key"],
)

st.success("App is running. Claude Haiku is initialized and ready for the next steps.")
