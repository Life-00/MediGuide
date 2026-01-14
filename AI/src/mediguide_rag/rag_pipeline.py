# rag_pipeline.py
import os
from dotenv import load_dotenv

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

store = {}

# 전역 설정
# .env 에 UPSTAGE_API_KEY 를 넣어두면 ChatUpstage / UpstageEmbeddings 가 자동으로 사용합니다.
# (필요하면 생성자에 upstage_api_key="..." 로 직접 넣어도 됩니다.)
PERSIST_DIR = "./chroma_db_fixed"

# Solar Pro 2 모델 ID (Upstage Console 문서 기준)
SOLAR_PRO2_MODEL = os.getenv("UPSTAGE_SOLAR_MODEL", "solar-pro2")
# 임베딩 모델 ID (langchain-upstage 권장)
SOLAR_EMBED_MODEL = os.getenv("UPSTAGE_EMBED_MODEL", "solar-embedding-1-large")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    """Main.py에서 검색기만 따로 쓰기 위함"""
    embeddings = UpstageEmbeddings(model=SOLAR_EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def get_rag_chain():
    # 1) 임베딩 & DB 설정
    embeddings = UpstageEmbeddings(model=SOLAR_EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2) LLM 설정: Upstage Solar Pro 2
    # - params/옵션은 langchain-upstage 버전에 따라 달라질 수 있어,
    #   가장 안전한 형태(temperature 등) 위주로 넣었습니다.
    # - IBM watsonx의 stop_sequences 같은 필드는 Upstage 쪽에서 이름/지원 여부가 다를 수 있습니다.
    llm = ChatUpstage(
        model=SOLAR_PRO2_MODEL,
        temperature=0.0,
        max_tokens=2000,
        # 필요 시 아래처럼 추가 파라미터를 kwargs로 넣는 패턴을 쓰기도 합니다.
        # (버전별 지원이 다를 수 있음)
        # top_p=0.9,
    )

    # 3) 프롬프트
    template = """
# Role
You are the **Senior AI Lawyer** for 'MediGuide'.
Your task is to analyze legal precedents for the user. To maximize readability, you must utilize **Markdown Formatting** perfectly.

# Critical Constraints (Data Source Principles)
1. **Source of Truth:** Construct the case information using **ONLY** the content provided in [CONTEXT].
2. **No Hallucination:** NEVER mix the facts from the [QUERY] (User's current story) with the facts from the [CONTEXT] (Past legal precedent). They are distinct entities.
3. **Reference:** If the [CONTEXT] is empty, honestly state that the search failed.
4. **Language:** Process the instructions in English, but **YOU MUST GENERATE THE FINAL RESPONSE IN KOREAN.**

# Output Style Guidelines (Visualization)
1. **Tables:** You must use a **Markdown Table** to compare 'Patient's Claim' vs. 'Hospital's Claim'.
2. **Highlights:** Use blockquotes (`>`) and bold text (`**`) to emphasize the final verdict/result.
3. **Structure:** Use emojis (🔍, 📋, ⚖️) to clearly distinguish sections.

# Output Format (Strictly follow this structure)

## 1. 🔍 검색된 유사 판례
(Write the case title and a one-sentence summary based on [CONTEXT])

## 2. 📋 판례 팩트 체크 (Fact Check)
**※ 주의: 아래는 사용자의 사연이 아닌, 검색된 과거 판례 데이터입니다.**
* **환자 정보:** (Summary of patient info from [CONTEXT])
* **사건 경위:** (Chronological flow of the incident from [CONTEXT])
* **병원 조치:** (Medical actions taken by the hospital from [CONTEXT])

## 3. ⚔️ 양측 주장 비교 (쟁점)
* **환자 측:** (Patient/Applicant's claim from [CONTEXT])
* **병원 측:** (Hospital/Respondent's claim from [CONTEXT])

## 4. ⚖️ 법원의 최종 판단
> **"[결과: 인정/불인정]"**
> (Summarize the verdict and reasoning from [CONTEXT] within 3 lines)

## 5. 💡 메디가이드 솔루션 (For User)
(Here, connect the user's [QUERY] with the analyzed [CONTEXT] to provide advice)
사용자님의 질문인 **"[Key point of QUERY]"**과 관련하여, 위 판례는 다음 시사점을 줍니다.

* **✅ 승소/대응 전략:** (Strategic advice based on the precedent)
* **📝 필수 확보 증거:** (List specific medical records or evidence to secure)

---
[INPUT DATA]
[HISTORY]: {chat_history}
[CONTEXT]: {context}
[QUERY]: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # retriever.invoke 결과가 "Document 리스트"로 들어오므로,
    # LLM이 읽기 좋게 텍스트로 합쳐주는 처리를 해주는 게 안정적입니다.
    def format_docs(docs):
        if not docs:
            return ""
        return "\n\n".join(
            [
                f"[{i+1}] {getattr(d, 'page_content', str(d))}"
                for i, d in enumerate(docs)
            ]
        )

    chain = (
        RunnableMap(
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
        )
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
    llm = ChatUpstage(
        model=SOLAR_PRO2_MODEL,
        temperature=0.0,
        max_tokens=2000,
    )

    legal_template = """
# Role
당신은 '메디가이드(MediGuide)'의 수석 AI 변호사입니다.
당신의 목표는 사용자 상담 내역을 분석하여, 법적 효력이 있는 **[의료분쟁 조정신청서(내용증명)]** 초안을 완벽하게 작성하는 것입니다.

# Input Data
[상담 내역]: {chat_history}

# Workflow (작업 절차)
당신은 반드시 아래 단계에 따라 사고하고 행동해야 합니다.

### 단계 1: 필수 정보 확인 (The Core 4)
상담 내역에 다음 4가지 핵심 정보가 모두 포함되어 있는지 확인하십시오.
1. **피해자(발신인):** 누구의 사고인가? (이름을 모르면 'OOO'으로 표기 가능)
2. **대상 병원(수신인):** 어느 병원인가? (병원이름을 모르면 질문 필요)
3. **사고 내용(Fact):** 어떤 시술/수술을 했고, 어떤 피해(결과)가 발생했는가?
4. **요구 사항(Demand):** 금전적 보상을 원하는가? (구체적 금액이 없으면 공란으로 둠)

### 단계 2: 모드 결정 (Decision)

**[모드 A: 인터뷰 (정보 부족)]**
만약 **'대상 병원'**이나 **'사고 내용'** 자체가 없다면, 문서를 작성하지 말고 사용자에게 질문하십시오.
* **제약:** 질문은 **최대 3개**까지만 하십시오.
* **출력 예시:** "완벽한 문서 작성을 위해 다음 2가지 정보가 필요합니다. 1. 병원 이름이 무엇인가요? 2. 정확히 어떤 수술을 받으셨나요?"

**[모드 B: 문서 작성 (정보 충분)]**
핵심 정보가 파악되었다면, 즉시 문서를 작성하십시오.
* **형식:** 아래 [작성 템플릿]의 구조를 엄격히 따르십시오.
* **빈칸 처리:** 주소나 정확한 날짜 등 사소한 정보가 빠져있다면, 묻지 말고 `(주소 기재)`, `(날짜)`와 같이 괄호로 표시하여 사용자가 나중에 채우도록 하십시오.
* **법률 용어 변환:** 사용자의 일반적인 표현을 전문 법률 용어로 변환하여 작성하십시오.
    * 실수했다 -> **"술기상 주의의무 위반"**
    * 신경 안 써줬다 -> **"경과관찰 의무 위반"**
    * 몰랐다/말 안 해줬다 -> **"설명 의무 위반"**

---

# Reference Examples (참고용 스타일 및 톤앤매너)

/** 예시 1: 내과 (대장내시경 중 천공 발생) **/
(사용자가 제공한 예시 1 내용 참고 - 생략하지 말고 모델이 참고하도록 내부적으로 인지함)
- 주장: 시술상의 주의의무 위반, 경과관찰 의무 위반

/** 예시 2: 내과 (심근경색 오진) **/
(사용자가 제공한 예시 2 내용 참고)
- 주장: 진단상의 과실(오진), 전원 의무 위반

/** 예시 3: 정형외과 (수술 중 혈관 손상) **/
(사용자가 제공한 예시 3 내용 참고)
- 주장: 술기상 과실, 혈관 손상

---

# Output Template (작성 양식)
**※ 주의: 반드시 Markdown 형식을 사용하고, 가독성을 위해 단락 사이에 줄바꿈을 넣으십시오.**

**발신인:** [이름]
**수신인:** [병원명] 대표원장 (또는 담당의사)
**주  소:** (병원 주소 기재)
**제  목:** [날짜] [시술명] 중 발생한 의료과실에 따른 손해배상 청구의 건

**1. 귀 병원의 무궁한 발전을 기원합니다.**

**2. 당사자 관계**
발신인은 귀 병원에서 [시술/수술명]을 받은 환자(혹은 보호자)이며, 수신인은 해당 의료행위를 시행하고 관리할 책임이 있는 의료기관입니다.

**3. 사건의 경위 (사실 관계)**
* 발신인은 [날짜] 귀 병원에 내원하여 [시술명]을 시행받았습니다.
* [상담 내역에 기반한 구체적인 사고 발생 경위 서술]
* [그로 인한 현재 피해 상태 및 악결과 서술]

**4. 발신인의 주장 (책임의 근거)**
**가. [법적 과실명 1: 예) 술기상 주의의무 위반]**
[해당 과실에 대한 논리적 설명]

**나. [법적 과실명 2: 해당될 경우만 작성]**
[설명]

**5. 요청 사항**
위와 같은 귀 병원의 과실로 발신인은 [피해 내용 요약]라는 회복하기 어려운 손해를 입었습니다.
이에 기왕 치료비 및 위자료를 합산한 **금 [금액]원**을 **[답변 기한: 작성일로부터 2주 뒤 날짜]**까지 배상해 줄 것을 청구합니다.

만약 기한 내 합리적인 답변이 없을 시, 한국의료분쟁조정중재원 조정 신청 및 민·형사상 법적 절차를 진행하겠습니다.

**[작성일]** [오늘 날짜]
**발신인:** [이름] (인)
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
    # 분류는 작은 모델(빠른/저렴)을 쓰는 게 일반적이지만,
    # Upstage 쪽에 별도 소형 분류 모델이 없다면 동일 모델로도 동작합니다.
    llm = ChatUpstage(
        model=SOLAR_PRO2_MODEL,
        temperature=0.0,
        max_tokens=5,
    )

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
