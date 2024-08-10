import streamlit as st
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import io

# Load API key from external file
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()
    
# Replace with your OpenAI API key
api_key_path = 'api_key.txt'
api_key = load_api_key(api_key_path)
os.environ["OPENAI_API_KEY"] = api_key


def load_documents(file):
    """
    Load documents from a PDF file.
    """

    loader = PyMuPDFLoader(file)
    return loader.load()

# Streamlit App
st.title("Question Answering with Langchain")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Load documents from the uploaded file
    save_directory = "uploaded_files"
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, uploaded_file.name)

    # Write the file to the specified path
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    documents = load_documents(save_path)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector store
    vector = FAISS.from_documents(documents, embeddings)

    # Create prompt
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    # Create LLM
    llm = ChatOpenAI(temperature=0.0)  # Adjust temperature as needed

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    def answer_question(query):
        """
        This function takes a question as input and returns the answer retrieved from the chain.
        """
        response = retrieval_chain.invoke({"input": query})
        return response["answer"]

    # User input for question
    user_question = st.text_input("Ask a question about the document:", key="user_input")

    if user_question:
        answer = answer_question(user_question)
        st.write("Answer:", answer)


# To run
# py -3 -m streamlit run "C:\AIDI\SEM2\KNOWLEDGE AND EXP\ASSIGNMENT2\create_stuff_documents_chain.py"
# 
