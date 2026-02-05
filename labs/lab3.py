import streamlit as st
from openai import OpenAI
import tiktoken

# Show title and description.
st.title("MY Lab 3 question answering chatbot")

# Max tokens to pass to the LLM (token-based buffer limit).
max_tokens = 1000

openAI_model = st.sidebar.selectbox("Which Model?",
                         ("mini", "regular"))
if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

# Token counting for chat messages (same encoding family as gpt-4o / gpt-4o-mini).
try:
    encoding = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages):
    """Total number of tokens for OpenAI chat messages."""
    total = 3  # reply priming
    for m in messages:
        total += 4
        total += len(encoding.encode(m["role"]))
        total += len(encoding.encode(m.get("content", "") or ""))
    return total
# System prompt: answer like for a 10-year-old and follow the "more info?" flow.
KID_FRIENDLY_SYSTEM = (
    "You explain things in a simple, friendly way so that a 10-year-old can understand. "
    "Use short sentences and everyday words. When you answer a question, give a clear answer "
    "and then ask: 'Do you want more info?' When the user wants more info, give more details "
    "on the same topic in the same simple style, then ask again: 'Do you want more info?'"
)

# Conversation phase: "ask_question" | "answered_ask_more" | "gave_more_ask_again"
if "phase" not in st.session_state:
    st.session_state.phase = "ask_question"
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

def is_yes(text):
    t = (text or "").strip().lower()
    return t in ("yes", "y", "yeah", "yep", "sure", "more", "please")

def is_no(text):
    t = (text or "").strip().lower()
    return t in ("no", "n", "nope", "nah", "no thanks")

# Create an openAI client
if "client" not in st.session_state:
    api_key = st.secrets["openai_api_key"]
    st.session_state.client = OpenAI(api_key=api_key)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "What would you like to know? Ask me anything!"}
    ]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("Enter a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client
    phase = st.session_state.phase
    last_question = st.session_state.last_question

    # ---- User said "No" (after we asked "Do you want more info?") → back to help ----
    if phase in ("answered_ask_more", "gave_more_ask_again") and is_no(prompt):
        reply = "Sure! What else can I help you with?"
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.phase = "ask_question"
        st.session_state.last_question = ""
    # ---- User said "Yes" (want more info) → provide more, then ask again ----
    elif phase in ("answered_ask_more", "gave_more_ask_again") and is_yes(prompt):
        # Build messages for LLM: system + recent context so it can give more on the same topic.
        messages_for_llm = [{"role": "system", "content": KID_FRIENDLY_SYSTEM}]
        # Keep recent conversation so the model knows the topic.
        messages_for_llm.extend(st.session_state.messages[-6:])  # last few exchanges
        messages_for_llm.append({
            "role": "user",
            "content": "The user said they want more information. Give more details about what we were just talking about, in the same simple way. Then end by asking: Do you want more info?",
        })
        messages_to_send = messages_for_llm
        if count_tokens(messages_to_send) > max_tokens:
            messages_to_send = [messages_to_send[0]] + messages_to_send[-4:]
        tokens_this_request = count_tokens(messages_to_send)
        st.caption(f"Tokens sent to LLM: {tokens_this_request} / {max_tokens}")
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages_to_send,
            stream=True,
        )
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.phase = "gave_more_ask_again"
    # ---- New question (or unclear reply): answer then ask "Do you want more info?" ----
    else:
        if phase == "ask_question" or not (is_yes(prompt) or is_no(prompt)):
            st.session_state.last_question = prompt
        messages_for_llm = [{"role": "system", "content": KID_FRIENDLY_SYSTEM}]
        messages_for_llm.extend(st.session_state.messages)
        while count_tokens(messages_for_llm) > max_tokens and len(messages_for_llm) > 3:
            # Keep system + last exchanges
            if messages_for_llm[1]["role"] == "assistant":
                messages_for_llm = [messages_for_llm[0]] + messages_for_llm[3:]
            else:
                messages_for_llm = [messages_for_llm[0]] + messages_for_llm[2:]
        tokens_this_request = count_tokens(messages_for_llm)
        st.caption(f"Tokens sent to LLM: {tokens_this_request} / {max_tokens}")
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages_for_llm,
            stream=True,
        )
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.phase = "answered_ask_more"

    # Token-based buffer: trim message history for next turn
    messages = st.session_state.messages
    while count_tokens(messages) > max_tokens and len(messages) > 2:
        if messages[0]["role"] == "assistant" and len(messages) > 3:
            messages = [messages[0]] + messages[3:]
        else:
            messages = messages[2:]
    st.session_state.messages = messages