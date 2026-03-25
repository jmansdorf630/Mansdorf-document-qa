import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("Lab 6 — Movie recommendations")

if "anthropic_api_key" not in st.secrets:
    st.error("Add `anthropic_api_key` to `.streamlit/secrets.toml` for Claude.")
    st.stop()

llm = init_chat_model(
    "claude-haiku-4-5-20251001",
    model_provider="anthropic",
    api_key=st.secrets["anthropic_api_key"],
)

if "last_recommendation" not in st.session_state:
    st.session_state.last_recommendation = None

with st.sidebar:
    st.header("Movie match")
    genre = st.selectbox(
        "Genre",
        (
            "Action",
            "Comedy",
            "Horror",
            "Drama",
            "Sci-Fi",
            "Thriller",
            "Romance",
        ),
    )
    mood = st.selectbox(
        "Mood",
        (
            "Excited",
            "Happy",
            "Sad",
            "Bored",
            "Scared",
            "Romantic",
            "Curious",
            "Tense",
            "Melancholy",
        ),
    )
    persona = st.selectbox(
        "Persona",
        (
            "Film Critic",
            "Casual Friend",
            "Movie Journalist",
        ),
    )

prompt = PromptTemplate.from_template(
    """You are helping someone pick what to watch.

Persona (adopt this voice, vocabulary, and attitude in your entire reply): {persona}
Genre they want: {genre}
Mood they are in: {mood}

Recommend exactly 3 movies that fit the genre and mood.
Write in the style of the persona above—match its tone (formal vs. casual, analytical vs. chatty, etc.).
For each movie, give a short reason it fits their mood and genre choice.
Use clear formatting (e.g. numbered list)."""
)

chain = prompt | llm | StrOutputParser()

st.caption(
    "Pick genre, mood, and persona in the sidebar, then run the chain. "
    "Try the same picks with different personas to compare tone."
)

if st.button("Get movie recommendations"):
    with st.spinner("Asking Claude…"):
        st.session_state.last_recommendation = chain.invoke(
            {"genre": genre, "mood": mood, "persona": persona}
        )

if st.session_state.last_recommendation:
    st.markdown(st.session_state.last_recommendation)
