import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings # to get embedding models
from langchain_core.documents import Document # to store text and metadata
from langchain_text_splitters import CharacterTextSplitter # to split the large paragraph into small chunks
from langchain_community.vectorstores import FAISS # to store the embedding data from the given document for similarity search

key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)
model=genai.GenerativeModel('gemini-2.5-flash-lite')
def load_embedding():
  return HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

with st.spinner('Loading embedding model.. ⌛'):
  embedding_model = load_embedding()

st.set_page_config('RAG DEMO', page_icon='🎯', layout='wide')
st.title('RAG Assistant :blue[Using Embedding and LLM] 🤖🧠🇦🇮👾')
st.subheader(':green[Your Intelligent Document Assistant ֎🇦🇮]')

uploaded_file = st.file_uploader('Upload file here in PDF format 🗂️', type=['pdf'])

if uploaded_file:
  pdf = PdfReader(uploaded_file)
  raw_text = ''
  
  for page in pdf.pages:
    raw_text += page.extract_text()
    
  if raw_text.strip(): # remove spaces and check whether have text data and ensures the given raw_text is not empty
    doc = Document(page_content=raw_text) 
    
    splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    
    # max char in each chunk is 1000 and overlap to maintain the relation between context = 200
    
    chunk_text = splitter.split_documents([doc])
    # splits the data in document
    
    text = [ i.page_content for i in chunk_text]
    # to get data as list of smaller text
    
    vector_db = FAISS.from_texts(text, embedding_model)
    retrieve = vector_db.as_retriever()
    # create a search tool to find the relevant chunks
    
    st.success('Document processed and saved successfully!!✅ Ask a question now🤔❓')
    query=st.text_input('ask me a question ?')
    
    if query:
        with st.chat_message('human'):
            
            with st.spinner('analysing the document...'):
                
                relevant_data=retrieve.invoke(query)
                #invoke the embedding model and search the similiar chunk in faiss for the given query
                content='\n\n'.join([ i.page_content for i in relevant_data])
                    
                prompt=f''' 
                you are an ai expert use the generated content {content}
                to answer the query {query}. if you are not sure with the answer say 
                " have no content related to this question,please ask relevant questions"
                result in bullet points '''
                
                response=model.generate_content(prompt)
                
                st.markdown("## :green[Results]")
                st.write(response.text)
    
    else:
        st.warning("drop the file in pdf format")

