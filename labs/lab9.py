import json
import os
import re

import streamlit as st
from openai import OpenAI

_MEMORIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memories.json")

MAIN_MODEL = "gpt-4o-mini"
EXTRACTION_MODEL = "gpt-4.1-nano"

BASE_SYSTEM = (
    "You are a helpful, friendly assistant. Be concise unless the user asks for detail. "
    "When long-term memories about the user are listed below, use them naturally in your "
    "replies so the conversation feels continuous."
)


def load_memories():
    """Load memories from memories.json; return [] if the file does not exist."""
    if os.path.exists(_MEMORIES_FILE):
        with open(_MEMORIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(m) for m in data if m]
        return []
    return []


def save_memories(memories):
    """Write a list of memories to memories.json."""
    with open(_MEMORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)


def build_system_prompt(memories):
    if not memories:
        return BASE_SYSTEM
    lines = "\n".join(f"- {m}" for m in memories)
    return (
        f"{BASE_SYSTEM}\n\n"
        "Here are things you remember about this user from past conversations:\n"
        f"{lines}"
    )


def parse_json_fact_list(raw_text):
    """Parse a JSON array of strings from the model; return [] on failure."""
    text = (raw_text or "").strip()
    if not text:
        return []
    # Strip ```json ... ``` fences if present
    fence = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def extract_new_memories(client, existing_memories, user_message, assistant_message):
    """Ask a small model for new user facts as a JSON list; no duplicates vs existing."""
    existing_json = json.dumps(existing_memories, ensure_ascii=False)
    extraction_user = f"""Analyze this conversation turn. Identify any NEW facts about the user worth remembering long-term
(name, preferences, interests, location, job, hobbies, goals, etc.).

Existing memories already saved (do NOT repeat or paraphrase these; only add genuinely new information):
{existing_json}

User message:
{user_message}

Assistant message:
{assistant_message}

Return ONLY a JSON array of strings. Each string is one concise fact.
If there is nothing new to remember, return [].
Example: ["User prefers dark mode", "User lives in Seattle"]

No markdown, no explanation—only the JSON array."""

    resp = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": extraction_user}],
    )
    return parse_json_fact_list(resp.choices[0].message.content)


st.title("Chatbot with Long-Term Memory")
st.write(
    "Chat normally—the bot loads saved memories into its system prompt and, after each "
    "reply, tries to learn new facts about you and store them in **memories.json**."
)

if "openai_api_key" not in st.secrets:
    st.error("Add `openai_api_key` to `.streamlit/secrets.toml`.")
    st.stop()

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["openai_api_key"])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I remember what we save between sessions. What’s on your mind?",
        }
    ]

with st.sidebar:
    st.header("Memories")
    memories = load_memories()
    if not memories:
        st.caption("No memories yet. Start chatting!")
    else:
        for mem in memories:
            st.markdown(f"- {mem}")
    if st.button("Clear all memories"):
        save_memories([])
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    memories = load_memories()
    system_prompt = build_system_prompt(memories)
    messages_for_api = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    client = st.session_state.client
    stream = client.chat.completions.create(
        model=MAIN_MODEL,
        messages=messages_for_api,
        stream=True,
    )
    with st.chat_message("assistant"):
        assistant_text = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    try:
        new_facts = extract_new_memories(
            client, memories, prompt, assistant_text
        )
    except Exception:
        new_facts = []
    updated = list(memories)
    for fact in new_facts:
        if fact not in updated:
            updated.append(fact)
    if updated != memories:
        save_memories(updated)
