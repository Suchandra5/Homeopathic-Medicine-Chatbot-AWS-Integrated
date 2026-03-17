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

from langchain_classic.chains.retrieval_qa.base import RetrievalQA


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

bucket_name = "homeopathy-datasets"
file_key = "homeopathic_medicines_dataset_1700.csv"


# Download file from S3
obj = s3.get_object(Bucket=bucket_name, Key=file_key)
# Load dataset
df = pd.read_csv(io.BytesIO(obj["Body"].read()))

df = df.fillna("")
# Convert dataset → documents
documents = []

for _, row in df.iterrows():

    text = row["case_description"]  # use RAG optimized column

    documents.append(Document(page_content=text))


# Embedding model (unchanged)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Build or load FAISS index
if os.path.exists("faiss_homeopathy_index"):
    
    vectorstore = FAISS.load_local(
        "faiss_homeopathy_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_homeopathy_index")


# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}   # slightly better retrieval
)


# Gemini model (unchanged)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)


# RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# Chat loop
while True:

    question = input("\nAsk your medical question: ")

    if question.lower() == "exit":
        break

    result = qa.invoke({"query": question})

    print("\n" + "="*60)
    print("🩺 Homeopathic Assistant")
    print("="*60)

    print(result["result"])

    print("="*60)