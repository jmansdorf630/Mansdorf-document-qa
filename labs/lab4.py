import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import zipfile
from pypdf import PdfReader
from io import BytesIO


def create_lab4_vectordb():
    """
    Construct a ChromaDB collection named "Lab4Collection" with PDF documents.
    Uses OpenAI embeddings model (text-embedding-3-small).
    Stores the collection in st.session_state.Lab4_VectorDB to avoid recreating it.
    """
    # Check if vector database already exists in session state
    if "Lab4_VectorDB" in st.session_state:
        return st.session_state.Lab4_VectorDB
    
    # Initialize OpenAI client
    if "client" not in st.session_state:
        api_key = st.secrets["openai_api_key"]
        st.session_state.client = OpenAI(api_key=api_key)
    
    client = st.session_state.client
    
    # Create OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["openai_api_key"],
        model_name="text-embedding-3-small"
    )
    
    # Initialize ChromaDB client (persistent storage)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get the collection
    try:
        collection = chroma_client.get_collection(
            name="Lab4Collection",
            embedding_function=openai_ef
        )
        # Collection exists, check if it has documents
        if collection.count() > 0:
            st.session_state.Lab4_VectorDB = collection
            return collection
    except Exception:
        # Collection doesn't exist, create it
        collection = chroma_client.create_collection(
            name="Lab4Collection",
            embedding_function=openai_ef
        )
    
    # Path to the zip file
    zip_path = "/Users/jakemansdorf/Downloads/Lab-04-Data.zip"
    
    # List of PDF files to process
    pdf_files = [
        "IST 488 Syllabus - Building Human-Centered AI Applications.pdf",
        "IST 314 Syllabus - Interacting with AI.pdf",
        "IST 343 Syllabus - Data in Society.pdf",
        "IST 256 Syllabus - Intro to Python for the Information Profession.pdf",
        "IST 387 Syllabus - Introduction to Applied Data Science.pdf",
        "IST 418 Syllabus - Big Data Analytics.pdf",
        "IST 195 Syllabus - Information Technologies.pdf"
    ]
    
    # Extract and process PDFs from zip file
    documents = []
    metadatas = []
    ids = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for pdf_filename in pdf_files:
            # Construct the full path within the zip
            zip_internal_path = f"Lab-04-Data/{pdf_filename}"
            
            try:
                # Read PDF from zip
                pdf_bytes = zip_ref.read(zip_internal_path)
                pdf_reader = PdfReader(BytesIO(pdf_bytes))
                
                # Extract text from all pages
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                # Clean up text (remove excessive whitespace)
                text_content = " ".join(text_content.split())
                
                if text_content.strip():  # Only add if text was extracted
                    documents.append(text_content)
                    metadatas.append({
                        "filename": pdf_filename,
                        "source": "Lab-04-Data"
                    })
                    ids.append(pdf_filename)  # Use filename as unique ID
                    
            except KeyError:
                st.warning(f"PDF file not found in zip: {pdf_filename}")
                continue
            except Exception as e:
                st.error(f"Error processing {pdf_filename}: {str(e)}")
                continue
    
    # Add documents to ChromaDB collection
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        st.session_state.Lab4_VectorDB = collection
        st.success(f"Successfully created ChromaDB collection with {len(documents)} documents!")
    else:
        st.error("No documents were successfully processed from the PDF files.")
        return None
    
    return collection


# --- Lab 4 page UI ---
st.title("Lab 4 – Document vector database")

st.write(
    "This page loads the Lab 4 ChromaDB collection from the 7 syllabus PDFs. "
    "The collection is created once and reused for the session."
)

# Initialize the vector DB when the page loads (only runs if not already in session_state)
vectordb = create_lab4_vectordb()

if vectordb is not None:
    count = vectordb.count()
    st.info(f"Vector DB ready: **Lab4Collection** has {count} document(s).")
else:
    st.warning(
        "Vector DB could not be loaded. Check that the Lab-04-Data.zip file is available "
        "at the expected path and that the OpenAI API key is set in secrets."
    )