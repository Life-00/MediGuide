# rag_pipeline.py (Production-grade patch: Anti-hallucination + Anti-infinite-interview + Safe fallback + A/B/C)
import os
import json
import re
from typing import List, Tuple, Dict, Any, Optional

from dotenv import load_dotenv

from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_chroma import Chroma

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

load_dotenv()

# ---------------------------------------------------------------------
# Global memory store (session -> chat history)
# ---------------------------------------------------------------------
store: Dict[str, ChatMessageHistory] = {}

# (문진 무한 루프 방지용) session -> interview turn count
_interview_turns: Dict[str, int] = {}

# ---------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------
IBM_URL = os.getenv("IBM_CLOUD_URL")
PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API = os.getenv("API_KEY")

PERSIST_DIR = "./chroma_db_fixed"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "mediguide_cases")

# Model IDs (env로 교체 가능)
MAIN_LLM_ID = os.getenv("MAIN_LLM_ID", "meta-llama/llama-3-405b-instruct")
RERANK_LLM_ID = os.getenv("RERANK_LLM_ID", "ibm/granite-3-8b-instruct")
ROUTER_LLM_ID = os.getenv("ROUTER_LLM_ID", "ibm/granite-3-8b-instruct")
WRITER_LLM_ID = os.getenv("WRITER_LLM_ID", "meta-llama/llama-3-405b-instruct")

# ---------------------------------------------------------------------
# Retrieval knobs (A: score gate / B: rerank)
# ---------------------------------------------------------------------
CANDIDATE_K = 25
FINAL_K = 5

# distance(낮을수록 유사) 가정. 환경에 따라 튜닝 필요.
MAX_DISTANCE_THRESHOLD = float(os.getenv("MAX_DISTANCE_THRESHOLD", "0.45"))

MAX_CONTEXT_CHARS_PER_DOC = 1400

# 문진 최대 턴(세션 당)
MAX_INTERVIEW_TURNS = int(os.getenv("MAX_INTERVIEW_TURNS", "2"))

# ---------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    if session_id not in _interview_turns:
        _interview_turns[session_id] = 0
    return store[session_id]


# ---------------------------------------------------------------------
# Embeddings + VectorStore
# ---------------------------------------------------------------------
def _build_embeddings() -> WatsonxEmbeddings:
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    return WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url=IBM_URL,
        project_id=PROJECT_ID,
        params=embed_params,
        apikey=WATSONX_API,
    )


def _build_vectorstore(embeddings: WatsonxEmbeddings) -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# ---------------------------------------------------------------------
# (B) Re-ranker LLM
# ---------------------------------------------------------------------
def _build_rerank_llm() -> WatsonxLLM:
    return WatsonxLLM(
        model_id=RERANK_LLM_ID,
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 120,
            "min_new_tokens": 1,
            "repetition_penalty": 1.0,
            "stop_sequences": ["\n\n", "</s>", "<|end_of_text|>"],
        },
    )


def answer_with_sources(question: str, session_id: str = "default_user") -> Dict[str, Any]:
    """
    main.py에서 '근거 불일치'를 없애기 위한 단일 진실 소스.
    - rag_chain이 만든 최종 답변과,
    - 그 답변에 사용된 최종 rerank docs를 함께 반환.

    return:
      {
        "answer": str,
        "mode": "SOLUTION"|"INTERVIEW",
        "docs": List[Document]
      }
    """
    # ✅ 세션 히스토리 확보
    history = get_session_history(session_id)

    # ✅ 여기서 get_rag_chain()을 매번 새로 만들면 느려질 수 있으니
    #    기존 main.py에서 생성한 rag_chain을 쓰는 게 이상적이지만,
    #    구조상 여기서는 체인을 한 번 생성해서 사용.
    chain = get_rag_chain()

    # chain은 내부적으로 retrieval_step에서 mode/docs/context를 만들고,
    # route_and_answer에서 문자열을 반환한다.
    #
    # BUT: 현재 get_rag_chain() 구현은 최종적으로 "문자열"만 반환하고 있어 docs를 밖으로 못 꺼냄.
    # 그래서 아래 방식으로 "동일 로직"을 한 번 더 수행해서 docs를 만들고, 답변은 chain을 사용한다.
    #
    # (최고 완성형은 get_rag_chain 내부에서 docs를 함께 반환하도록 구조를 바꾸는 것)
    embeddings = _build_embeddings()
    vectorstore = _build_vectorstore(embeddings)
    rerank_llm = _build_rerank_llm()

    # 1) 후보 검색 + 게이트
    pairs = _retrieve_candidates_with_scores(vectorstore, question, k=CANDIDATE_K)
    docs = [d for d, _ in pairs]
    scores = [s for _, s in pairs]

    if not _passes_gate(scores):
        mode = "INTERVIEW"
        final_docs: List[Document] = []
    else:
        mode = "SOLUTION"
        final_docs = _rerank_docs(rerank_llm, question, docs, top_n=FINAL_K)

    # 2) 답변 생성(세션 메모리 업데이트는 RunnableWithMessageHistory가 수행)
    answer = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )

    return {"answer": answer, "mode": mode, "docs": final_docs}
# ---------------------------------------------------------------------
# Main answer LLM
# ---------------------------------------------------------------------
def _build_main_llm() -> WatsonxLLM:
    return WatsonxLLM(
        model_id=MAIN_LLM_ID,
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 900,
            "min_new_tokens": 10,
            "repetition_penalty": 1.08,
            "stop_sequences": ["<|end_of_text|>", "\n\n질문:", "User:"],
        },
    )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _norm_text(x: str) -> str:
    if x is None:
        return ""
    x = str(x).replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def _safe_int_list_from_json(text: str) -> List[int]:
    text = (text or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(i, int) for i in data):
            return data
    except Exception:
        pass

    nums = re.findall(r"\d+", text)
    return [int(n) for n in nums][:FINAL_K]


def _format_docs_for_context(docs: List[Document]) -> str:
    """
    (C) [근거 n] 포맷 강제. case_id 노출 금지.
    """
    if not docs:
        return ""

    blocks = []
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "제목 없음")
        dept = d.metadata.get("dept", d.metadata.get("medical_dept", "진료과 없음"))
        section = d.metadata.get("section", "section 없음")
        seq = d.metadata.get("seq", "")

        body = _norm_text(d.page_content or "")
        if len(body) > MAX_CONTEXT_CHARS_PER_DOC:
            body = body[:MAX_CONTEXT_CHARS_PER_DOC] + "..."

        header = f"[근거 {i}] 사건명: {title} | 진료과: {dept} | 섹션: {section}"
        if seq:
            header += f" | 원문번호: {seq}"

        blocks.append(f"{header}\n{body}")

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------
# (A) Score-gated retrieval + (B) rerank
# ---------------------------------------------------------------------
def _retrieve_candidates_with_scores(
    vectorstore: Chroma, query: str, k: int = CANDIDATE_K
) -> List[Tuple[Document, float]]:
    return vectorstore.similarity_search_with_score(query, k=k)


def _passes_gate(scores: List[float]) -> bool:
    if not scores:
        return False
    return min(scores) <= MAX_DISTANCE_THRESHOLD


def _rerank_docs(
    rerank_llm: WatsonxLLM, query: str, docs: List[Document], top_n: int = FINAL_K
) -> List[Document]:
    if not docs:
        return []

    snippets = []
    for idx, d in enumerate(docs):
        title = d.metadata.get("title", "제목 없음")
        section = d.metadata.get("section", "section 없음")
        dept = d.metadata.get("dept", d.metadata.get("medical_dept", ""))
        text = _norm_text(d.page_content or "")
        text = text[:500] + ("..." if len(text) > 500 else "")
        snippets.append(f"{idx}. (사건명: {title} | 진료과: {dept} | 섹션: {section}) {text}")

    rerank_prompt = f"""
당신은 검색 결과 재정렬(rerank) 모델입니다.

사용자 질문과 가장 관련성이 높은 문서 인덱스 {top_n}개를 골라,
반드시 JSON 배열 형식으로만 출력하세요. (예: [3, 0, 7, 2, 1])

규칙:
- 부가 설명 금지
- JSON 배열만 출력
- 인덱스는 중복 없이
- 길이가 {top_n}보다 짧아도 되지만, 가능한 한 {top_n}개를 선택

[사용자 질문]
{query}

[후보 문서 목록]
{chr(10).join(snippets)}
""".strip()

    raw = rerank_llm.invoke(rerank_prompt)
    picks = _safe_int_list_from_json(raw)

    seen = set()
    valid = []
    for i in picks:
        if 0 <= i < len(docs) and i not in seen:
            valid.append(i)
            seen.add(i)
        if len(valid) >= top_n:
            break

    if not valid:
        valid = list(range(min(top_n, len(docs))))

    return [docs[i] for i in valid]


# ---------------------------------------------------------------------
# Public API: retriever only (main.py sources 카드용)
# ---------------------------------------------------------------------
def get_retriever():
    embeddings = _build_embeddings()
    vectorstore = _build_vectorstore(embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 25},
    )


# ---------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------
def get_rag_chain():
    embeddings = _build_embeddings()
    vectorstore = _build_vectorstore(embeddings)
    rerank_llm = _build_rerank_llm()
    llm = _build_main_llm()

    # =========================================================
    # (C) Prompts (솔루션/문진 + 안전장치)
    # =========================================================
    system_template = """
# Identity
당신은 '메디가이드(MediGuide)'의 20년 경력 의료소송 전문 변호사 역할입니다.
사용자는 법·의학 지식이 없는 일반인입니다.

# Safety / Anti-hallucination (최우선)
- 당신은 실제 로펌이나 기관이 아닙니다.
- 전화번호, 웹사이트, 주소, 담당자 등 실존 연락처를 절대 만들어내지 마세요.
- 대리 협상, 소송 수행 등 현실 세계의 행위를 직접 수행한다고 주장하지 마세요.
- 법원·대법원·판결문·판례를 언급할 때는 반드시 [Context]의 근거가 있을 때만 사용하세요.
- 근거가 없는 경우 "판례"라는 표현을 사용하지 마세요.

# Evidence-First (근거 우선)
- 모든 사실 판단은 제공된 [Context]에 근거해야 합니다.
- [Context]에 없는 사실(날짜, 병원명, 금액, 진단명 등)은 단정하지 마세요.
- 다만 절차, 증거 수집, 일반적인 다음 단계 안내는 일반 원칙으로 제시할 수 있습니다.
  이 경우 반드시 "일반적인 안내입니다."라는 표현을 한 문장 포함하세요.

# Forbidden
- 내부 식별자(case_id, chunk_id) 노출 금지. 사건명(title)만 사용하세요.
- 회피성 표현("경우에 따라", "단정하기 어렵다"만 반복)을 금지합니다.
- 근거가 있는 범위에서는 과거형 단정 문장으로 서술하세요.

# Citation Rules (매우 중요)
- [Context]의 근거는 [근거 1] ~ [근거 N] 형식입니다.
- 솔루션 모드에서는 아래 각 섹션(1~5)마다 최소 1회 이상 [근거 n]을 인용하세요.
- 근거가 없는 문장은 작성하지 마세요.

# Output Constraints
- 각 섹션은 2~3문장 이내로 간결하게 작성하세요.
- 모든 문장은 마침표로 끝내세요.
- 불필요한 반복을 금지합니다.

# Required Format (형식 고정)
### 1. 🔍 검색된 유사 사례 요약
### 2. 📋 사건의 전말 (Fact)
### 3. ⚔️ 환자와 병원의 주장 (Issues)
### 4. ⚖️ 최종 판단 결과 (Result)
### 5. 💡 메디가이드 Action Plan

# Decision Rule (판단 강제 규칙)
- 아래 조건 중 3개 이상 충족 시 반드시 의료과실 인정 가능성에 대해 명확히 판단하세요.
  1) 시술·검사명이 특정되고 날짜가 확인됨.
  2) 시술 직후 또는 직접적 인과관계가 있는 신체 손상이 발생함.
  3) 사전 설명 또는 동의 과정에 해당 위험이 포함되지 않음.
  4) 의료기관이 책임을 부인하거나 무대응 상태임.
- 판단은 "의료과실이 인정될 가능성이 있다/낮다" 중 하나로 명시하세요.
- 이 규칙은 [Context]가 충분한 솔루션 모드에서만 적용합니다.

# Analogy Rule (유사성 판단)
- 동일한 시술이 아니더라도 의료행위 중 발생한 신체 손상 사례는 유사 판례로 분석할 수 있습니다.
- 특히 내시경 중 치아 손상, 기구 삽입 중 신체 손상, 마취 중 부작용 사례는
  동일한 법리(주의의무, 설명의무)로 판단하세요.

# Output Hygiene (출력 위생)
- 단독 기호(".", "-", "*")만 있는 줄을 출력하지 마세요.
- "질문:", "답변:" 같은 라벨을 사용하지 마세요.
- 연속된 빈 줄(2줄 이상)을 만들지 마세요.
- 항상 완성된 문장으로 시작하세요.

# Action Plan Rules
- 지금 당장 할 일을 3~6개 bullet로 제시하세요.
- 증거 수집을 최우선으로 하세요
  (진료기록, 동의서, 검사결과, 치과 소견서, 견적서, 사진, 녹취, 메신저).
- 다음 절차(병원 공식 요청, 의료분쟁조정중재원 신청 등)를 1~2개 포함하세요.

""".strip()

    # 문진: "요." 같은 구어체/추임새 방지 + 질문 수 제한 + (왜 필요한지) 포함
    interview_system = """
# Identity
당신은 '메디가이드(MediGuide)'의 20년 경력 의료소송 전문 변호사 역할로 상담합니다.

# Identity Lock (정체성 고정)
- 사용자가 정체나 소개를 묻는 경우(예: "자기소개", "너 누구야", "메디가이드가 뭐야")에는
  아래 한 문장만 출력하고 종료하세요.
  "저는 메디가이드의 20년 경력 의료소송 전문 변호사입니다."
- AI, 챗봇, 시스템, 상담 AI라는 표현을 절대 사용하지 마세요.

# Strict Style
- 문장 시작에 감탄사, 추임새, 구어체("요.", "음,", "자,", "안녕하세요")를 사용하지 마세요.
- 과장된 자기소개를 금지합니다.
- 출력은 '공감 1문장 + 질문 3~5개'로만 구성하세요.

# Task (Smart Interview)
- 현재는 유사 판례 [Context]가 부족하거나 관련도가 낮습니다.
- 결론을 내리지 말고, 사실관계와 증거 확보에 필요한 질문만 하세요.
- 질문은 3~5개로 제한하세요.
- 각 질문 뒤에 (왜 필요한지) 1문장을 덧붙이세요.

# Question Focus (우선순위)
1) 시술·수술·검사명과 날짜(YYYY-MM-DD).
2) 피해 내용(현재 증상, 치료 경과, 치과 진단 여부).
3) 설명 및 동의 과정(설명의무)과 동의서 존재 여부.
4) 병원 대응(기록 제공, 보상, 회신 여부).
5) 확보 가능한 증거
   (진료기록, 동의서, 영수증, 치과 소견서, 사진, 녹취, 메신저).

# Output Hygiene (출력 위생)
- 단독 기호(".", "-", "*")만 있는 줄을 출력하지 마세요.
- "질문:", "답변:" 같은 라벨을 사용하지 마세요.
- 연속된 빈 줄(2줄 이상)을 만들지 마세요.
- 항상 완성된 문장으로 시작하세요.

""".strip()

    # 게이트 실패 후 문진도 끝났는데도 근거가 부족한 경우: "일반 가이드" 안전 출력
    fallback_system = """
# Identity
당신은 '메디가이드(MediGuide)'의 20년 경력 의료소송 전문 변호사 역할로 상담합니다.

# Situation
유사 판례 [Context]가 부족하여 특정 판례를 인용해 단정할 수 없습니다.

# Rules
- 판례/대법원/법원/판결을 언급하지 마세요.
- [Context] 없이 사실을 단정하지 마세요.
- 아래 형식으로만 출력하세요.

# Output Format (반드시 유지)
### 1. 현재 단계에서 가능한 판단 범위
### 2. 바로 확보해야 할 증거 (우선순위)
### 3. 병원에 요청할 문구(짧게)
### 4. 다음 절차(중재원/분쟁 조정) 체크리스트
""".strip()

    solution_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "질문: {question}\n\n[Context]\n{context}"),
        ]
    )

    interview_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", interview_system),
            MessagesPlaceholder("chat_history"),
            ("human", "질문: {question}"),
        ]
    )

    fallback_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", fallback_system),
            MessagesPlaceholder("chat_history"),
            ("human", "질문: {question}"),
        ]
    )

    # =========================================================
    # Retrieval step (A + B) + Interview-turn gate
    # =========================================================
    def retrieval_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        session_id = inputs.get("session_id", "default_user")

        pairs = _retrieve_candidates_with_scores(vectorstore, question, k=CANDIDATE_K)
        docs = [d for d, _ in pairs]
        scores = [s for _, s in pairs]

        has_good_context = _passes_gate(scores)

        if not has_good_context:
            # 게이트 실패: 우선 문진 모드 (단, 턴 제한)
            return {
                **inputs,
                "mode": "INTERVIEW",
                "docs": [],
                "context": "",
                "scores": scores,
                "session_id": session_id,
            }

        # 게이트 통과: rerank 후 context 구성
        reranked = _rerank_docs(rerank_llm, question, docs, top_n=FINAL_K)
        context = _format_docs_for_context(reranked)

        return {
            **inputs,
            "mode": "SOLUTION",
            "docs": reranked,
            "context": context,
            "scores": scores,
            "session_id": session_id,
        }

    def route_and_answer(inputs: Dict[str, Any]) -> str:
        mode = inputs.get("mode", "INTERVIEW")
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])
        context = inputs.get("context", "")
        session_id = inputs.get("session_id", "default_user")

        # 문진 턴 제한
        if mode == "INTERVIEW":
            _interview_turns[session_id] = _interview_turns.get(session_id, 0) + 1

            # 1~MAX_INTERVIEW_TURNS 까지는 문진
            if _interview_turns[session_id] <= MAX_INTERVIEW_TURNS:
                return (interview_prompt | llm | StrOutputParser()).invoke(
                    {"question": question, "chat_history": chat_history}
                )

            # 문진 턴 초과: 더 이상 질문 폭주 금지 → 일반 가이드로 전환
            return (fallback_prompt | llm | StrOutputParser()).invoke(
                {"question": question, "chat_history": chat_history}
            )

        # 솔루션 모드에서는 문진 턴 카운터 리셋(정상적으로 근거를 찾았다는 뜻)
        _interview_turns[session_id] = 0

        if context.strip():
            return (solution_prompt | llm | StrOutputParser()).invoke(
                {"question": question, "context": context, "chat_history": chat_history}
            )

        # 이론상 여기 오면 안 되지만, 안전장치
        return (fallback_prompt | llm | StrOutputParser()).invoke(
            {"question": question, "chat_history": chat_history}
        )

    base_chain = (
        RunnableMap(
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", []),
                # RunnableWithMessageHistory config에서 세션을 받기 때문에,
                # 여기서는 안전하게 기본값 처리만.
                "session_id": lambda x: x.get("session_id", "default_user"),
            }
        )
        | RunnableLambda(retrieval_step)
        | RunnableLambda(route_and_answer)
    )

    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


def get_writing_chain():
    """
    대화 내역 기반 문서 작성 전용 체인
    (주의: 문서 반복 출력 이슈는 main.py에서 history 정제/중복 저장을 먼저 잡는 게 핵심)
    """
    llm = WatsonxLLM(
        model_id=WRITER_LLM_ID,
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 2200,
            "min_new_tokens": 120,
            "repetition_penalty": 1.0,
        },
    )

    legal_template = """
# Identity
당신은 '메디가이드(MediGuide)'의 의료소송 문서작성 AI입니다.

# Safety / Anti-hallucination
- 아래 [상담 내역]에 없는 사실(날짜/병원명/금액/진단명/시술명/주소/연락처 등)을 절대 지어내지 마세요.
- 전화번호/웹사이트/주소를 만들어내지 마세요.
- 욕설/비난/명예훼손 표현은 정중하고 법률적 표현으로 순화하세요.

# Decision
- 필수 정보 4가지(A~D) 중 하나라도 없으면 문서를 작성하지 말고,
  "문서 작성을 위해 아래 정보가 추가로 필요합니다."로 시작하는 질문만 출력하세요.

# Required Information
A. 사고 일시(최소 YYYY-MM-DD) 및 병원/의료기관 명칭
B. 사건 경위(어떤 시술/검사/진료 중 무엇이 발생)
C. 피해 내용(현재 증상/치료 경과/추가 치료 여부/생활 불편)
D. 청구 금액(총액)

# Output Rules
- 마크다운 설명 금지. 문서 본문 텍스트만 출력.
- 중복 문장 반복 금지. 같은 문장을 2회 이상 반복하지 마세요.

# Document Template (정보 충분 시)
제목: 의료과실에 따른 손해배상(조정) 신청/청구의 건

1. 당사자
- 신청인(환자): [상담 내역에 있으면 기재, 없으면 공란]
- 피신청인(의료기관): [병원명], [주소: 있으면 기재, 없으면 공란]

2. 신청 취지
- 신청인은 피신청인에게 의료과실로 인한 손해배상으로 금 [총 청구금액]원을 지급할 것을 청구합니다.
- 지급 기한: 본 문서 수령일로부터 14일 이내.

3. 사건 개요(사실관계)
- (1) 진료/시술 경위:
- (2) 문제 발생 및 경과:
- (3) 현재 피해 상태:

4. 신청인의 주장(책임의 근거)
- (1) 주의의무 위반 정황:
- (2) 설명의무 위반 정황:
- (3) 인과관계 및 손해:

5. 손해 내역 및 청구 금액
- 치료비:
- 위자료:
- 합계:

6. 증거 자료(확보/예정)
- 진료기록부, 검사결과, 동의서, 영수증, 사진/영상, 메시지/통화 기록, 치과 소견서(해당 시)

7. 요청 사항
- 피신청인은 기한 내 서면으로 회신 바랍니다.

[작성일] [상담 내역에 있으면 반영, 없으면 공란]
신청인: __________________ (서명 또는 인)

---
[상담 내역]
{chat_history}
""".strip()

    prompt = ChatPromptTemplate.from_template(legal_template)

    return (
        {"chat_history": lambda x: x["chat_history"]}
        | prompt
        | llm
        | StrOutputParser()
    )


def get_router_chain():
    """
    DOC vs CHAT 분류
    """
    llm = WatsonxLLM(
        model_id=ROUTER_LLM_ID,
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 5,
            "min_new_tokens": 1,
        },
    )

    template = """
# Role
당신은 '메디가이드(MediGuide)'의 Intent Classifier입니다.

# Output Rules
- 출력은 오직 DOC 또는 CHAT 중 하나여야 합니다.
- 공백/줄바꿈/따옴표/마침표/설명/이모지 금지.
- 예: DOC

# DOC
- 내용증명/손해배상 청구서/조정신청서/합의서/공문/이메일 등 문서 작성 또는 수정 요청
- 금액/날짜/이름/항목 추가·삭제/톤 변경/양식 요구 포함

# CHAT
- 의료과실 상담, 판례 검색/해석, 절차/서류/증거 안내, 용어 설명, 감정 호소

# Tie-break
- DOC 신호가 1개라도 있으면 DOC.

# User Input
{question}
""".strip()

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

