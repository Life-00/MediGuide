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
    # 1. 임베딩 & 검색기 설정 (기존과 동일)
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

    # 2. LLM 설정 (70B 사용)
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-405b-instruct",
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 600,
            "min_new_tokens": 10
        }
    )

    # =========================================================
    # 🌟 [STEP 1] 질문 재구성 (Contextualize Query)
    # =========================================================
    condense_q_system_prompt = """
당신은 '의료 사고 검색 쿼리 최적화 전문가'입니다.
[채팅 내역]과 [사용자의 마지막 질문]을 분석하여, 오직 '내과 의료 분쟁 판례'를 검색하기 위한 단일 독립 질문(Standalone Question)으로 재구성하세요.

# 규칙:
1. 대명사(그거, 이 사건, 당시 등)를 구체적인 의료 용어(예: 대장내시경 천공, 직장암 오진 등)로 치환하세요.
2. 답변을 하지 마세요. 오직 검색 쿼리만 출력하세요.
3. 질문이 이미 완벽하다면 수정하지 말고 그대로 반환하세요.
"""
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )
    
    condense_q_chain = condense_q_prompt | llm | StrOutputParser()

    # =========================================================
    # 🌟 [STEP 2] 메인 답변 프롬프트 (수정됨)
    # =========================================================
    template =  """
# Role
당신은 '메디가이드(MediGuide)'의 수석 변호사입니다. 20년 경력의 의료 소송 전문성을 바탕으로, 사용자의 고통에 공감하되 법리적으로는 냉철하고 정확한 분석을 제공합니다.

# Context (검색된 판례 데이터)
{context}

# Instructions (엄격 준수)
1. **Fact-Only:** 반드시 주어진 [Context]의 내용에만 기반하여 답변하세요. 모르는 내용을 추측하거나 지어내면 법적 책임이 따를 수 있음을 명심하고, 데이터가 없다면 "유사한 판례를 찾지 못했습니다"라고 솔직하게 답한 뒤 '스마트 문진'으로 전환하세요.
2. **Structure:** - [의료적 공감]: 사용자의 상황을 요약하며 따뜻하게 위로합니다.
   - [법률 분석 및 판례 인용]: [Context]의 '사건명(title)'과 '사건번호(case_id)'를 언급하며, 해당 사건에서 '병원의 과실'이 왜 인정(또는 부정)되었는지 핵심 이유를 설명합니다.
   - [메디가이드 Action Plan]: 사용자가 당장 확보해야 할 증거(의료기록지, CCTV, 동의서 등)를 구체적으로 제시합니다.
3. **Tone:** 전문적이고 신뢰감 있는 한국어 문어체를 사용하세요.

# Response Format
[공감과 결론]: ...
[유사 판례 분석]: ...
[전문가의 조언]: ...

# User Question
{question}

# Answer
""" 
    # 👆 [수정] 위 template에서 {question} 부분은 제거했습니다. (아래 human 메시지와 중복 방지)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template), # 👈 [수정] prompt 대신 template 변수 사용
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )

    # =========================================================
    # 🌟 [STEP 3] 체인 연결
    # =========================================================
    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return chain_with_history



def get_writing_chain():
    """
    RAG 검색 없이 오직 '대화 내역'만 보고 내용증명서를 작성하는 전용 체인
    """
    # 405B 모델 권장 (지시 이행력이 가장 좋음)
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-405b-instruct", 
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy", 
            "max_new_tokens": 2000,
            "min_new_tokens": 100,
            "repetition_penalty": 1.0
        }
    )
    
    legal_template =  """
당신은 대한민국 최고의 의료 전문 법무 대필 AI입니다. 
사용자의 [상담 내역]을 바탕으로 법적 효력을 갖출 수 있는 '손해배상 청구 내용증명서'를 작성하세요.

# 🚨 CRITICAL RULES (위반 시 시스템 오류)
1. **No Chatting:** "작성해드렸습니다", "도움이 되길 바랍니다" 등 문서 외의 어떠한 설명도 절대 포함하지 마세요. 출력은 반드시 '발신인'으로 시작해서 '인'으로 끝나야 합니다.
2. **Data Integration:** {chat_history}에 이름, 날짜, 금액, 병원명이 있다면 괄호 [ ]를 제거하고 해당 정보를 직접 기입하세요. 정보가 없다면 [ ] 형태로 남겨두세요.
3. **Legal Tone:** '귀 병원의 무궁한 발전을 기원합니다', '엄중히 책임을 묻겠습니다' 등 실제 법률 문서에서 사용하는 격식 있는 표현을 사용하세요.

# 문서 서식 (내용증명)
--------------------------------------------------
내 용 증 명

발신인: [이름]
수신인: [병원명/원장명]
주  소: [병원 주소]

제  목: 의료과실에 따른 손해배상 청구 및 통지의 건

1. 귀 병원의 무궁한 발전을 기원합니다.

2. 당사자 관계 및 사건의 발생
   발신인은 [날짜] 귀 병원에서 [진료/수술명]을 받은 환자이며, 수신인은 해당 의료행위의 주체로서 환자에 대한 주의의무 및 설명의무를 지는 의료기관입니다.

3. 사실 관계 (사건 경위)
   - [상담 내역을 바탕으로 사건을 시나리오대로 재구성하여 작성]

4. 과실 및 법적 책임에 대한 주장
   - 본 사건은 의료진의 [과실 내용: 예시 - 설명 의무 위반, 주의 의무 위반]으로 인하여 발생한 명백한 사고입니다. 
   - 판례에 따르면 의료진은 발생 가능한 위험을 사전에 설명할 의무가 있으나, 본 건에서는 이를 소홀히 하였습니다.

5. 손해배상 청구 금액 및 요청 사항
   - 발신인은 위 사고로 인하여 [피해 내용]의 유무형적 손해를 입었습니다.
   - 이에 금 [청구 금액] 원의 배상을 청구하며, 본 통보서를 수령한 날로부터 14일 이내에 성의 있는 답변을 주시기 바랍니다.
   - 기한 내 원만한 합의가 이루어지지 않을 경우, 한국의료분쟁조정중재원 신청 및 민·형사상 법적 절차를 즉시 착수할 것임을 통지합니다.

[작성일] 2026년 01월 13일 (현재 날짜 적용)
발신인: [이름] (인)
--------------------------------------------------

[상담 내역]
{chat_history}
    """
    
    prompt = ChatPromptTemplate.from_template(legal_template)
    
    return (
        {"chat_history": lambda x: x["chat_history"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    
def get_router_chain():
    """
    사용자의 질문 의도를 'CHAT'(상담) 또는 'DOC'(문서작성/수정)으로 분류하는 체인
    """
    # 판단은 70B 모델 사용 
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-405b-instruct",
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy", # 확률 놀이 금지 (가장 확실한 답만 선택)
            "max_new_tokens": 5,         # 단어 딱 하나만 뱉게 제한
            "min_new_tokens": 1
        }
    )
    
    # [철벽 방어 프롬프트]
    template = """
당신은 '메디가이드'의 지능형 라우터입니다. 
사용자의 질문(Q)이 '단순 상담'인지 '문서 작성'인지 분류하여 오직 한 단어(DOC 또는 CHAT)만 출력하세요.

# 분류 가이드:
- **DOC**: "써줘", "작성해줘", "문서로 만들어줘", "방금 내용 정리해서 청구서 초안 짜줘", "날짜/이름/금액 수정해줘"와 같은 명시적 명령이 있을 때.
- **CHAT**: 의료 사고 상담, 판례 검색, 위로 요청, 일반적인 질문, 상황 설명 등 그 외 모든 경우.

# 예시:
Q: "수면내시경 사고 판례 알려줘" -> CHAT
Q: "지금까지 말한 거 내용증명으로 써줘" -> DOC
Q: "이름을 김철수로 바꿔서 다시 써줘" -> DOC

Q: {question}
A: """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt | llm | StrOutputParser()