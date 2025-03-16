import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_google_vertexai import VertexAIEmbeddings
#from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.chains import SequentialChain
from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain

with st.sidebar:
    st.title("Chat Bot ")
    st.markdown('''
    ## about
    This app is an Chatbot
        ''')
    st.write("Made by Chirag Kaushik")

def main():
    st.header("Chat with pdf")
    # upload a file
    pdf= st.file_uploader("Upload your pdf",type='pdf')
        
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)
        text =''
        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
                )
        chunks = text_splitter.split_text(text=text)
            # Embendding
            
    #st.write(chunks)
    
    
        embeddings = HuggingFaceEmbeddings()
        VectorStore=FAISS.from_texts(chunks, embedding=embeddings)
        
            #st.write(text)

    query=st.text_input("Ask any Question:")
    st.write(query)
    if query:
        docs=VectorStore.similarity_search(query=query,k=3)
        #st.write(docs)
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key='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',
            temperature=0.7,
        )
            
        chain = load_qa_chain(llm=llm , chain_type="stuff")
        response = chain.run(input_documents=docs, question = query)
        st.write(response)
if __name__ == '__main__':
    main()  
