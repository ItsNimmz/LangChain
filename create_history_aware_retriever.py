import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain import hub
from langchain_community.retrievers import WikipediaRetriever

# Pull the rephrase prompt from LangChain's hub
rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

# Load API key from external file
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()
    
# Replace with your OpenAI API key
api_key_path = 'api_key.txt'
api_key = load_api_key(api_key_path)

# Initialize the ChatOpenAI model
llm = ChatOpenAI(api_key=api_key)

retriever = WikipediaRetriever()

# Create the history-aware retriever chain
chat_retriever_chain = create_history_aware_retriever(
    prompt=rephrase_prompt,  # Provide the required prompt argument
    llm=llm,
    retriever=retriever
)

st.title("AI Chat with History-Aware Retriever")

user_input = st.text_input("Ask a question:", "What is AI?")
chat_history = st.text_area("Chat History:", "Tell me more about machine learning.")

if st.button("Get Response"):
    result = chat_retriever_chain.invoke({
        "input": user_input,
        "chat_history": chat_history.split("\n")
    })
    st.write(result)


# & py -3 -m streamlit run "C:\AIDI\SEM2\KNOWLEDGE AND EXP\ASSIGNMENT2\create_history_aware_retriever.py"