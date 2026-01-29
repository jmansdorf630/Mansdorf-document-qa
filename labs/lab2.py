import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("MY Document question answering")
st.write(
    "Upload a document below to generate a summary – GPT will summarize it based on your selected options!"
)

# Get the OpenAI API key from Streamlit secrets.
openai_api_key = st.secrets["openai_api_key"]

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Sidebar for summary options and model selection.
with st.sidebar:
    st.header("Summary Options")
    summary_type = st.radio(
        "Choose a summary type:",
        ("Summarize the document in 100 words", 
         "Summarize the document in 2 connecting paragraphs", 
         "Summarize the document in 5 bullet points")
    )
    use_advanced_model = st.checkbox("Use advanced model")

# Determine the model based on the checkbox.
model = "gpt-4" if use_advanced_model else "gpt-3.5-turbo"

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)", type=("txt", "md")
)

# If a file is uploaded, generate the summary.
if uploaded_file:
    document = uploaded_file.read().decode()
    messages = [
        {
            "role": "user",
            "content": f"{summary_type}: {document}",
        }
    ]

    # Generate the summary using the selected model.
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    # Stream the response to the app using `st.write_stream`.
    st.header("Document Summary")
    st.write_stream(stream)
