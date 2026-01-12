# rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

load_dotenv()

store = {}

# 전역 설정
IBM_URL = os.getenv('IBM_CLOUD_URL')
PROJECT_ID = os.getenv('PROJECT_ID')
WATSONX_API = os.getenv('API_KEY')
PERSIST_DIR = "./chroma_db_fixed"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    """Main.py에서 검색기만 따로 쓰기 위함"""
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    embeddings = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url=IBM_URL,
        project_id=PROJECT_ID,
        params=embed_params,
        apikey=WATSONX_API
    )
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={'k': 3})

def get_rag_chain():
    # 1. 임베딩 & DB 설정
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    embeddings = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url=IBM_URL,
        project_id=PROJECT_ID,
        params=embed_params,
        apikey=WATSONX_API
    )
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # 2. LLM 설정 
    llm = WatsonxLLM(
        
        model_id="meta-llama/llama-3-405b-instruct",
        
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 800,
            "min_new_tokens": 10,
            # 모델이 똑똑해졌으니 페널티를 살짝 낮춰도 됩니다 (1.1 -> 1.05)
            "repetition_penalty": 1.05, 
            "stop_sequences": ["<|end_of_text|>", "\n\n질문:", "User:"]
        }
    )

    # 3. 프롬프트 
    template = template = """
# Role
당신은 '의료 소송 전문 AI 변호사'입니다.
주어진 [Context]를 바탕으로 사용자에게 전문적이고 따뜻한 조언을 제공하세요.

# Constraints (엄격 준수)
1. **언어:** 반드시 **'자연스러운 한국어'**로만 답변하세요. (영어, 베트남어 등 외국어 절대 금지)
2. **사건 인용:** 참고한 판례의 'case_id'가 단순 숫자(예: 1, 4, 126)라면, 숫자를 말하지 말고 **'사건명(title)'**을 언급하세요. (예: "사건번호 4번" (X) -> "위암 오진 사건 사례" (O))
3. **법률 번호:** 만약 실제 법원 사건번호(예: 20xx가합xxxx)가 있다면 그것을 우선적으로 언급하세요.
4. **반복 금지:** 했던 말을 또 하거나, 불필요한 URL을 생성하지 마세요.

# Output Format
1. **[공감과 결론]:** (사용자의 상황에 깊이 공감하는 멘트로 시작)
2. **[유사 판례 분석]:** (가장 유사한 판례의 핵심 내용과 배상 판결 요약)
3. **[전문가의 조언]:** (필요한 증거 서류나 대처 방안 2~3가지)

# Context (참고 판례)
{context}

# Chat History
{chat_history}

# User Question
{question}

# Answer
"""
    
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableMap({
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return chain_with_history