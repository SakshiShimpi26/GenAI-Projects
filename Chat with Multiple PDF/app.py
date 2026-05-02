from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Reading PDF files uploaded by the user
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

# Converting the raw_text pdf data into smallers chunks
def get_chunks_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Store that chunks in FAISS database as embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Make a chain for Question and Answer 
def get_conversational_chain():
    prompt_template = """ 
        Answer the question in detail from the provided context. Make sure to provide all details.
        If the answer is not available, just say "Answer not available" and make sure not to provide 
        any incorrect information.\n\n
        Context: \n{context}\n
        Question: \n{question}\n
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user input and return the answer of the questions asked by the user in a detailed and proficient manner.
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

# Main function which contains the basic user input code and UI code that is developed using the streamlit library
def main():
    st.set_page_config(page_title="Chat with multiple PDFs")
    st.header("Chat with multiple PDFs using Gemini Pro")
    
    user_question = st.text_input("Ask questions from uploaded PDF files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("MENU")
        # Adding the parameter accept_multiple_files so as to take multiple PDF files as input from the user rather than taking just a single file
        pdf_docs = st.file_uploader("Upload PDF files and click Submit", type=["pdf"], accept_multiple_files=True)
        if st.button("SUBMIT"):
            with st.spinner("Processing your files..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks_text(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete! You can now ask questions.")

if __name__ == "__main__":
    main()
