import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 환경 변수 로드
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# PDF 처리 및 벡터 스토어 초기화 (캐싱 적용)
@st.cache_resource
def initialize_vectorstore():
    loader = PyMuPDFLoader('2024_KB_부동산_보고서_최종.pdf')
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embedding_model = OpenAIEmbeddings()
    faiss_index_save = 'faiss_index'
    
    if os.path.exists(faiss_index_save):
        vectorstore = FAISS.load_local(
            faiss_index_save,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )
        vectorstore.save_local(faiss_index_save)
    return vectorstore

# 대화 체인 초기화 (캐싱 적용)
@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """당신은 KB부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

    {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def get_trimmed_history(history):
        return history[-4:] if len(history) > 4 else history

    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: get_trimmed_history(x.get('chat_history', [])),
            context=lambda x: format_docs(retriever.invoke(x['question']))
        )
        | prompt
        | model
        | StrOutputParser()
    )
    
    # 세션별 히스토리 관리를 위한 딕셔너리
    history_db = {}

    def get_session_history(session_id: str):
        if session_id not in history_db:
            history_db[session_id] = ChatMessageHistory()
        return history_db[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key='question',
        history_messages_key='chat_history'
    )

def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 체인 로드
    chain_with_memory = initialize_chain()

    # 기존 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        # 사용자 메시지 표시 및 저장
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain_with_memory.invoke(
                    {'question': prompt},
                    {'configurable': {'session_id': 'streamlit_session'}}
                )
                st.markdown(response)
        
        # AI 응답 저장
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
