try:
    import streamlit as st
except ImportError:
    raise ImportError("streamlit is not installed. Install with: python3 -m pip install streamlit")

from openai import OpenAI
from pypdf import PdfReader

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
    "Upload a document (.txt, .md, or .pdf)", type=("txt", "md", "pdf")
)

# If a file is uploaded, generate the summary.
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        document = ""
        for page in reader.pages:
            document += page.extract_text()
    else:
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
