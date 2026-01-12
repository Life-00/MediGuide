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
            "repetition_penalty": 1.1, 
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
    
    legal_template = """
    당신은 '문서 작성 전문 AI'입니다. 
    당신의 유일한 임무는 사용자와의 [상담 내역]을 분석하여, 완벽한 포맷의 '의료사고 손해배상 청구 내용증명서'를 작성하는 것입니다.

    # 🚨 [치명적 경고 (Critical Rules)] - 절대 어기지 마세요.
    1. **설명 금지:** "이렇게 작성했습니다", "빈칸을 채우세요", "확인해주세요" 같은 사족을 **절대** 붙이지 마세요.
    2. **전체 출력:** 수정 요청이 들어오면, 수정된 부분만 보여주지 말고 **반드시 문서의 처음부터 끝까지 전체를 다시 출력**하세요.
    3. **빈칸 처리:** 정보가 없으면 괄호로 남기되, **사용자가 구체적인 내용(금액, 날짜, 이름 등)을 제시했다면 괄호 대신 그 내용을 반드시 기입하세요.** (우선순위 최상)

    # [표준 서식]
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
       귀 병원은 시술 전 [설명 의무]를 충실히 이행하지 않았거나, 시술 과정상 주의 의무를 위반한 과실이 있다고 판단됩니다. (상담 내용 중 판례나 근거가 있다면 여기에 요약)

    5. 요청 사항
       이에 본인은 귀 병원에 의료과실에 대한 명확한 해명과 합리적인 배상안을 [답변 기한: 2주 후 날짜]까지 서면으로 회신해 줄 것을 정중히 요청합니다. 만약 기한 내 답변이 없을 시, 한국의료분쟁조정중재원 조정 신청 또는 민사 소송 등 법적 절차를 진행할 것임을 통지합니다.

    [작성일] 2026년 [월] [일]
    발신인: [환자 이름] (인)
    ---

    [상담 내역]
    {chat_history}

    # 작성된 내용증명서 (아래에 문서 내용만 출력):
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
        model_id="meta-llama/llama-3-3-70b-instruct",
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
    당신은 사용자 의도를 분류하는 '엄격한 관리자'입니다.
    질문을 분석하여 'DOC' 또는 'CHAT' 중 하나만 출력하세요.

    # 🚨 분류 절대 기준 (Strict Rules)
    
    1. **DOC (문서 작성/수정 요청)**
       - 사용자가 문서를 **"만들어달라", "써달라", "작성해달라", "수정해달라"**고 명시적으로 명령한 경우에만 해당합니다.
       - 예: "내용증명서 써줘", "이 내용으로 문서 만들어", "날짜를 수정해줘"
    
    2. **CHAT (그 외 모든 상황)**
       - 질문, 상담, 하소연, 상황 설명, 법적 가능성 문의 등은 무조건 CHAT입니다.
       - **중요:** "이거 의료사고인가요?" 처럼 묻는 건 문서를 써달라는 게 아닙니다. -> CHAT
       - **중요:** "치아가 부러졌어요" 처럼 상황을 말하는 건 문서를 써달라는 게 아닙니다. -> CHAT

    # [Few-Shot 예시]
    Q: "수면내시경 하다가 이빨이 깨졌어. 이거 보상받을 수 있어?"
    A: CHAT

    Q: "너무 억울해요. 병원에서는 책임 없다고만 해요."
    A: CHAT

    Q: "설명 의무 위반 판례 좀 알려줘."
    A: CHAT

    Q: "위 내용을 바탕으로 내용증명서 초안 작성해줘."
    A: DOC

    Q: "날짜를 2025년으로 고쳐서 다시 써줘."
    A: DOC

    # 사용자 질문
    Q: {question}
    A: 
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt | llm | StrOutputParser()