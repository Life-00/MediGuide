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


# rag_pipeline.py (맨 아래에 추가)

def get_writing_chain():
    """
    RAG 검색 없이 오직 '대화 내역'만 보고 내용증명서를 작성하는 전용 체인
    """
    # [모델 선택] 문서 작성은 '지능'이 높아야 하므로 405B 사용 (안 되면 70B나 Mistral)
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-405b-instruct", 
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 2000, # 문서는 기니까 토큰 많이 줌
            "min_new_tokens": 100,
            "repetition_penalty": 1.05
        }
    )
    
    # [핵심] 변호사 표준 내용증명 양식 (One-Shot Learning)
    # AI에게 "이 빈칸 채우기 놀이를 해"라고 시키는 것과 같습니다.
    legal_template = """
    당신은 20년 경력의 의료 소송 전문 변호사입니다.
    사용자와의 [상담 내역]을 정밀 분석하여, 아래 [표준 서식]에 맞춰 '의료사고 손해배상 청구 내용증명서' 초안을 작성하세요.

    # 작성 원칙 (Strict Rules)
    1. **빈칸 추론:** 상담 내역에 있는 정보(증상, 병원 반응, 수술명 등)를 찾아 서식의 빈칸을 채우세요.
    2. **정보 부재 시:** 상담 내용에 없는 정보(구체적인 날짜, 의사 이름 등)는 억지로 지어내지 말고 `[수술 날짜]`, `[병원명]` 처럼 괄호로 남겨두세요. (사용자가 직접 채울 수 있게)
    3. **어조:** 감정적인 호소를 배제하고, 차갑고 논리적인 법률 문체("~함", "~바람", "~통지함")를 사용하세요.
    4. **형식 유지:** 아래 서식의 목차 번호와 구조를 절대 깨지 마세요.

    [표준 서식]
    ---
    발신인: [환자 이름 (모르면 '본인')]
    수신인: [병원장 또는 담당의사]
    주  소: [병원 주소]
    제  목: 의료과실에 따른 손해배상 청구의 건

    1. 귀 병원의 무궁한 발전을 기원합니다.

    2. 당사자 관계
       발신인은 귀 병원에서 [수술/시술명]을 시술받은 환자이며, 수신인은 해당 의료행위를 시행한 의료기관입니다.

    3. 사건의 경위 (사실 관계)
       - 발신인은 [날짜] 귀 병원에 내원하여 [진단명] 진단을 받고 [수술/시술]을 진행하였습니다.
       - 그러나 시술 직후 [구체적인 부작용/증상]이 발생하였습니다.
       - 이에 대해 귀 병원 측은 [병원 측의 대응 내용]라고 답변하였으나, 이는 납득하기 어렵습니다.

    4. 발신인의 주장 (과실 내용)
       귀 병원은 시술 전 [설명 의무]를 충실히 이행하지 않았거나, 시술 과정상 주의 의무를 위반한 과실이 있다고 판단됩니다. (상담 내용 중 판례나 근거가 있다면 여기에 한 줄 요약)

    5. 요청 사항
       이에 본인은 귀 병원에 의료과실에 대한 명확한 해명과 합리적인 배상안을 [답변 기한: 2주 후 날짜]까지 서면으로 회신해 줄 것을 정중히 요청합니다. 만약 기한 내 답변이 없을 시, 한국의료분쟁조정중재원 조정 신청 또는 민사 소송 등 법적 절차를 진행할 것임을 통지합니다.

    [작성일] 2026년 [월] [일]
    발신인: [환자 이름] (인)
    ---

    [상담 내역]
    {chat_history}

    # 작성된 내용증명서:
    """
    
    prompt = ChatPromptTemplate.from_template(legal_template)
    
    return (
        {"chat_history": lambda x: x["chat_history"]}
        | prompt
        | llm
        | StrOutputParser()
    )