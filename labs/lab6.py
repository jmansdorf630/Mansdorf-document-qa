import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("Lab 6 — Movie recommendations")

if "openai_api_key" not in st.secrets:
    st.error("Add `openai_api_key` to `.streamlit/secrets.toml`.")
    st.stop()

llm = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    api_key=st.secrets["openai_api_key"],
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

followup_prompt = PromptTemplate.from_template(
    """You previously recommended these movies to the user:

---
{recommendations}
---

Answer their follow-up question clearly and helpfully. Stay grounded in the movies above;
you may add brief general film context if it helps.

User question:
{question}"""
)

followup_chain = followup_prompt | llm | StrOutputParser()

st.caption(
    "Pick genre, mood, and persona in the sidebar, then run the chain. "
    "Try the same picks with different personas to compare tone."
)

if st.button("Get movie recommendations"):
    with st.spinner("Asking OpenAI…"):
        st.session_state.last_recommendation = chain.invoke(
            {"genre": genre, "mood": mood, "persona": persona}
        )

if st.session_state.last_recommendation:
    st.markdown(st.session_state.last_recommendation)

    st.divider()
    follow_up = st.text_input("Ask a follow-up question about these movies:")
    if st.button("Submit follow-up question") and follow_up.strip():
        with st.spinner("Thinking…"):
            followup_answer = followup_chain.invoke(
                {
                    "recommendations": st.session_state.last_recommendation,
                    "question": follow_up.strip(),
                }
            )
        st.markdown("**Follow-up answer**")
        st.markdown(followup_answer)
