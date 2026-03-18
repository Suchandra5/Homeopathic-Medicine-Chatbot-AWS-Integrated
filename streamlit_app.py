import streamlit as st
import pandas as pd
import os
import io
import boto3
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

load_dotenv()

st.set_page_config(
    page_title="Homeopathic Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Homeopathic Medicine Chatbot")
st.caption("Powered by Gemini + FAISS + AWS S3")

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

GOOGLE_API_KEY        = get_secret("GOOGLE_API_KEY")
AWS_ACCESS_KEY_ID     = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_secret("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = get_secret("AWS_REGION")

st.write("AWS_ACCESS_KEY_ID loaded:", "YES" if AWS_ACCESS_KEY_ID else "NO")
st.write("AWS_SECRET_ACCESS_KEY loaded:", "YES" if AWS_SECRET_ACCESS_KEY else "NO")
st.write("AWS_REGION loaded:", "YES" if AWS_REGION else "NO")
st.write("GOOGLE_API_KEY loaded:", "YES" if GOOGLE_API_KEY else "NO")

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_qa_chain():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    bucket_name = "homeopathy-datasets-1"
    file_key    = "homeopathic_medicines_dataset_1700.csv"

    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df  = pd.read_csv(io.BytesIO(obj["Body"].read())).fillna("")

    documents = [
        Document(page_content=row["case_description"])
        for _, row in df.iterrows()
    ]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    index_path = "faiss_homeopathy_index"
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(index_path)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

qa_chain = load_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe your symptoms or ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# import streamlit as st
# import pandas as pd
# import os
# import io
# import boto3
# from dotenv import load_dotenv
# import warnings
# warnings.filterwarnings("ignore")

# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_classic.chains import RetrievalQA

# load_dotenv()

# # ── Page config ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="Homeopathic Assistant",
#     page_icon="🩺",
#     layout="centered"
# )

# st.title("🩺 Homeopathic Medicine Chatbot")
# st.caption("Powered by Gemini + FAISS + AWS S3")

# # ── Secrets ──────────────────────────────────────────────────
# def get_secret(key):
#     try:
#         return st.secrets[key]
#     except Exception:
#         return os.getenv(key)

# GOOGLE_API_KEY        = get_secret("GOOGLE_API_KEY")
# AWS_ACCESS_KEY_ID     = get_secret("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = get_secret("AWS_SECRET_ACCESS_KEY")
# AWS_REGION            = get_secret("AWS_REGION")

# # ── Debug: check if secrets are loading ──
# st.write("AWS_ACCESS_KEY_ID loaded:", "YES" if AWS_ACCESS_KEY_ID else "NO ❌")
# st.write("AWS_SECRET_ACCESS_KEY loaded:", "YES" if AWS_SECRET_ACCESS_KEY else "NO ❌")
# st.write("AWS_REGION loaded:", "YES" if AWS_REGION else "NO ❌")
# st.write("GOOGLE_API_KEY loaded:", "YES" if GOOGLE_API_KEY else "NO ❌")get_secret("AWS_REGION")

# # ── Load model + index once (cached) ─────────────────────────
# @st.cache_resource(show_spinner="Loading knowledge base...")
# def load_qa_chain():
#     s3 = boto3.client(
#         "s3",
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_REGION
#     )

#     bucket_name = "homeopathy-datasets-1"
#     file_key    = "homeopathic_medicines_dataset_1700.csv"

#     obj = s3.get_object(Bucket=bucket_name, Key=file_key)
#     df  = pd.read_csv(io.BytesIO(obj["Body"].read())).fillna("")

#     documents = [
#         Document(page_content=row["case_description"])
#         for _, row in df.iterrows()
#     ]

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     index_path = "faiss_homeopathy_index"
#     if os.path.exists(index_path):
#         vectorstore = FAISS.load_local(
#             index_path, embeddings, allow_dangerous_deserialization=True
#         )
#     else:
#         vectorstore = FAISS.from_documents(documents, embeddings)
#         vectorstore.save_local(index_path)

#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0.7,
#         google_api_key=GOOGLE_API_KEY
#     )

#     qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
#     return qa

# qa_chain = load_qa_chain()

# # ── Chat history ──────────────────────────────────────────────
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display previous messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # ── Input ─────────────────────────────────────────────────────
# if prompt := st.chat_input("Describe your symptoms or ask a question..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             result = qa_chain.invoke({"query": prompt})
#             answer = result["result"]
#         st.markdown(answer)

#     st.session_state.messages.append({"role": "assistant", "content": answer})
