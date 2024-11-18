import os
from typing import List, Dict
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

def create_embeddings_from_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> tuple:
    """Create embeddings from text using SentenceTransformers and FAISS"""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_text(text)

    # Create embeddings using SentenceTransformers
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store with improved parameters
    vectorstore = FAISS.from_texts(
        texts,
        embeddings,
        metadatas=[{"chunk": i} for i in range(len(texts))],
        normalize_L2=True
    )
    
    return vectorstore, texts

def setup_qa_chain(vectorstore):
    """Set up the question-answering chain with improved retrieval"""
    llm = ChatGroq(
        temperature=0.3,
        model_name="mixtral-8x7b-32768",
        max_tokens=4096,
        top_p=0.9
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 5,  # Number of documents to retrieve
                "fetch_k": 10,  # Number of documents to fetch before filtering
                "lambda_mult": 0.7  # Diversity of results (0.0-1.0)
            }
        ),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain

def get_response(qa_chain, question: str) -> Dict:
    """Get response from the QA chain"""
    result = qa_chain({"question": question})
    return {
        "answer": result["answer"],
        "source_documents": result["source_documents"]
    }

def generate_summary(text: str, max_length: int = 500) -> str:
    """Generate a summary of the text using Groq"""
    llm = ChatGroq(temperature=0.3, model_name="mixtral-8x7b-32768")
    prompt = f"""Please provide a concise summary of the following text in no more than {max_length} characters. 
    Focus on the main points and key takeaways:
    
    {text}
    """
    response = llm.predict(prompt)
    return response

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract key phrases and topics from the text"""
    llm = ChatGroq(temperature=0.3, model_name="mixtral-8x7b-32768")
    prompt = f"""Please extract the {max_keywords} most important keywords or key phrases from the following text.
    Return them as a comma-separated list:
    
    {text}
    """
    response = llm.predict(prompt)
    return [keyword.strip() for keyword in response.split(",")]

def analyze_sentiment(text: str) -> Dict:
    """Analyze the sentiment of the text"""
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    prompt = """Please analyze the sentiment of the following text. 
    Return a JSON-like response with these fields:
    - overall_sentiment: (positive, negative, or neutral)
    - confidence: (a number between 0 and 1)
    - brief_explanation: (1-2 sentences explaining why)
    
    Text to analyze:
    {text}
    """
    response = llm.predict(prompt.format(text=text))
    return eval(response)  # Convert string response to dict
