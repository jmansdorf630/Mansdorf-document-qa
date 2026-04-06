import base64
import streamlit as st
from openai import OpenAI

VISION_PROMPT = (
    "Describe the image in at least 3 sentences. Write five different captions for "
    "this image. Captions must vary in length, minimum one word but be no longer than "
    "2 sentences. Captions should vary in tone, such as, but not limited to funny, "
    "intellectual, and aesthetic."
)

st.title("Lab 8 — Image captioning (URL & upload)")
st.write(
    "Use **Part A** with a public image URL, or **Part B** by uploading an image. "
    "OpenAI’s vision API describes the image and suggests captions. "
    "Your OpenAI key comes from Streamlit secrets."
)

if "openai_api_key" not in st.secrets:
    st.error("Add `openai_api_key` to `.streamlit/secrets.toml`.")
    st.stop()

if "url_response" not in st.session_state:
    st.session_state.url_response = None
if "last_image_url" not in st.session_state:
    st.session_state.last_image_url = None

if "upload_response" not in st.session_state:
    st.session_state.upload_response = None
if "last_upload_bytes" not in st.session_state:
    st.session_state.last_upload_bytes = None

# --- Part A: Image URL ---
st.subheader("Part A: Image URL")
st.caption(
    "Use a URL that **points directly to an image file** (e.g. ends in .jpg, .png, or a "
    "CDN link that returns an image). Page URLs or HTML pages can cause API errors."
)

url = st.text_input("Image URL")

if st.button("Generate description and captions (URL)") and url:
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    )
    st.session_state.url_response = response.choices[0].message.content
    st.session_state.last_image_url = url

if st.session_state.url_response:
    st.markdown("**Part A — result**")
    if st.session_state.last_image_url:
        st.image(st.session_state.last_image_url)
    st.write(st.session_state.url_response)

# --- Part B: File upload ---
st.divider()
st.subheader("Part B: File upload")
st.write(
    "Upload a local image file. The image is sent as a base64 data URI with **detail: low** "
    "to reduce cost and tokens."
)

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "gif"],
)

if st.button("Generate description and captions (upload)") and uploaded:
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    b64 = base64.b64encode(uploaded.read()).decode("utf-8")
    mime = uploaded.type
    data_uri = f"data:{mime};base64,{b64}"
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "low"},
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    )
    st.session_state.upload_response = response.choices[0].message.content
    st.session_state.last_upload_bytes = uploaded.getvalue()

if st.session_state.upload_response:
    st.markdown("**Part B — result**")
    if st.session_state.last_upload_bytes is not None:
        st.image(st.session_state.last_upload_bytes)
    st.write(st.session_state.upload_response)
