from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os

loader=PyMuPDFLoader('2024_KB_부동산_보고서_최종.pdf')

documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=text_splitter.split_documents(documents)

print('분할된 청크 수:',len(chunks))

embedding_model=OpenAIEmbeddings()

faiss_index_save='faiss_index'

#기존 인덱스 로드 
#os(파이썬 기본 라이브러리로 컴퓨터 파일이나 폴더 다룰 떄 사용함)모듈은 기존에 파일이 이미 존재하는지를 확인하는 용도
if os.path.exists(faiss_index_save):
    vectorstore=FAISS.load_local(
        faiss_index_save,
        embedding_model,
        allow_dangerous_deserialization=True
        )
else:
    vectorstore=FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )
    vectorstore.save_local(faiss_index_save)
    print('신규 생성 성공')
    

print('문서의 수:',vectorstore.index.ntotal)

retriever=vectorstore.as_retriever(search_kwargs={"k":3})
template="""당신은 KB부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

{context}"""
prompt=ChatPromptTemplate.from_messages(
    [
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
    ]
)
model=ChatOpenAI(model_name='gpt-4o-mini',temperature=0)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

#최근 대화 4개 기록 가져옴.
def get_trimmed_history(history):
    return history[-4:] if len(history) > 4 else history

chain=(
    RunnablePassthrough.assign(
        chat_history=lambda x: get_trimmed_history(x.get('chat_history', [])),
        context=lambda x: format_docs(retriever.invoke(x['question']))
    )
    | prompt
    | model
    | StrOutputParser()
)

history_db = {} #세션별 저장 분리용 교체

def get_session_history(session_id:str):
    if session_id not in history_db:
        history_db[session_id]=ChatMessageHistory()
    return history_db[session_id]

chain_with_memory=RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='question',
    history_messages_key='chat_history'
    )

def chat_with_bot():
    session_id="user_session"
    print('KB 부동산 보고서 챗봇입니다. 질문해 주세요.(종료하려면 "quit" 입력)')
    while True:
        user_input=input("사용자: ")
        if user_input.lower()=='quit':
            break
        
        response=chain_with_memory.invoke(
            {'question': user_input},
            {'configurable': {'session_id': session_id}}
        )

        print("챗봇:", response)

    
if __name__ == "__main__":
    chat_with_bot()
