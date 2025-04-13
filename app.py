import json
import os
import sys
import boto3
import streamlit as st

# We will be using Titan Embeddings Model TO Generate Embedding
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain_community.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and Vector Store
def get_vector_store(docs):
    vector_store = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store.save_local("faiss_index")

def get_claude_llm():
    # Create the Anthropic Model
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={'max_tokens_to_sample': 512})
    return llm

def get_llama3_llm():
    # Create the Llama3 Model
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Gunakan potongan-potongan konteks berikut 
untuk memberikan jawaban ringkas terhadap 
pertanyaan di akhir, tetapi gunakan minimal 250 
kata dengan penjelasan yang rinci. Jika Anda tidak 
tahu jawabannya, katakan saja bahwa Anda tidak tahu, jangan mencoba untuk membuat jawaban yang tidak benar.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def delete_all_pdfs():
    folder = "data"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) and filename.endswith('.pdf'):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Gagal menghapus file {file_path}. Error: {e}")

def check_pdf_limit(folder="data", limit=5):
    files = [f for f in os.listdir(folder) if f.endswith('.pdf')]
    return len(files) >= limit

def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with PDF using AWS Bedrock 💁")

    if check_pdf_limit():
        st.warning("Jumlah file PDF di folder 'data' telah mencapai batas maksimum (5 file). Anda tidak dapat mengunggah file baru.")
    else:
        # Upload PDF files
        uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            os.makedirs("data", exist_ok=True)
            if len(uploaded_files) + len([f for f in os.listdir("data") if f.endswith('.pdf')]) > 5:
                st.error("Mengunggah file ini akan melebihi batas maksimum (5 file). Silahkan Hapus File PDF terlebih dahulu.")
            else:
                for uploaded_file in uploaded_files:
                    with open(os.path.join("data", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("PDF files have been uploaded and saved to 'data' folder.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

        if st.button("Delete All PDFs"):
            delete_all_pdfs()
            st.success("All PDF files have been deleted from the 'data' folder.")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
