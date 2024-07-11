import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever


# Load environemnt variables
load_dotenv()
persist_directory = str(Path(__file__).parent / 'chroma/')
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)


# Function to compute the hash of the document content
def compute_document_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()


# Process and store document chunks
def process_document(file_path, document_id):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )
    splits = text_splitter.split_documents(pages)
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=os.path.join(persist_directory, document_id)
    )
    return vectordb


# Statefully manage chat history
store = {}

def get_query_response(file_path, query, session_id):
    document_id = compute_document_hash(file_path=file_path)

    if is_document_processed(document_id=document_id):
        vectordb = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=os.path.join(persist_directory, document_id))
    else:
        vectordb = process_document(file_path=file_path, document_id=document_id)
    retriever = vectordb.as_retriever()

    # Contextualizer question
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=question_answer_chain
    )

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    # response = conversational_rag_chain.invoke(input={'input': query}, config={'configurable': {'session_id': session_id}})
    response = conversational_rag_chain.invoke(input={'input': query}, config={'configurable': {'session_id': session_id}})
    return response['answer']


# Check if the document is already processed
def is_document_processed(document_id):
    return os.path.exists(os.path.join(document_id, document_id))


# Run the program
def main():
    file_path = Path(__file__).parent / 'document.pdf'
    query = 'What this document is about?'
    if query:
        print(f'You: {query}')
        response = get_query_response(file_path=file_path, query=query, session_id='abc123')
        print(response)

if __name__ == '__main__':
    main()