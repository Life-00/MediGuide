# main.py (완성형 리팩토링 v2: answer_with_sources 강제, 세션 전달 확실화, DOC 반복/증식 방지, 스키마 통일)
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage


# -------------------------------------------------------------------------
# [Import] rag_pipeline.py에서 필요한 체인/유틸 가져오기
#  - 이 main.py는 answer_with_sources()가 "필수"입니다. (근거 불일치/중복검색 방지)
# -------------------------------------------------------------------------
try:
    from rag_pipeline import (
        get_rag_chain,
        get_writing_chain,
        get_router_chain,
        get_session_history,
        store,
        answer_with_sources,  # ✅ 필수
    )
except Exception as e:
    try:
        from src.mediguide_rag.rag_pipeline import (
            get_rag_chain,
            get_writing_chain,
            get_router_chain,
            get_session_history,
            store,
            answer_with_sources,  # ✅ 필수
        )
    except Exception as e2:
        raise RuntimeError(
            "rag_pipeline에서 answer_with_sources를 import 할 수 없습니다.\n"
            "✅ rag_pipeline.py에 answer_with_sources(question: str, session_id: str) -> dict\n"
            "   {'answer': str, 'mode': 'SOLUTION'|'INTERVIEW', 'docs': List[Document], 'sources': List[dict] (optional)}\n"
            "를 반드시 추가하세요.\n"
            f"import error1={e}\nimport error2={e2}"
        )


# -------------------------------------------------------------------------
# [Setup] FastAPI 앱 초기화
# -------------------------------------------------------------------------
app = FastAPI(title="MediGuide AI Server", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 데모 OK. 운영이면 도메인 제한 권장.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# [Loading] AI 모델 체인 로드
# -------------------------------------------------------------------------
print("🚀 AI 모델 체인 로딩 중...")
rag_chain = get_rag_chain()
writing_chain = get_writing_chain()
router_chain = get_router_chain()
print("✅ 로딩 완료! 서버가 준비되었습니다.")


# -------------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------------
MAX_QUERY_CHARS = 2500  # 프론트에서도 제한 권장

class Question(BaseModel):
    query: str = Field(..., description="User input")
    session_id: str = Field("default_user", description="Session identifier")

class SourceItem(BaseModel):
    evidence_no: Optional[int] = None
    title: str
    dept: str
    section: str
    seq: Optional[str] = None
    content_preview: str

class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    type: str  # "chat" | "document"
    answer: str
    document_content: Optional[str] = None
    mode: Optional[str] = None  # "SOLUTION" | "INTERVIEW"
    sources: List[SourceItem] = []
    latency_ms: int


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _sanitize_session_id(session_id: str) -> str:
    s = (session_id or "").strip()
    if not s:
        return "default_user"
    if len(s) > 64:
        s = s[:64]
    return s

def _sanitize_query(q: str) -> str:
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="query가 비어 있습니다.")
    if len(q) > MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"query가 너무 깁니다. (최대 {MAX_QUERY_CHARS}자)",
        )
    return q

def _safe_preview(text: str, n: int = 220) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) > n:
        return t[:n] + "..."
    return t

def _build_sources_from_docs(docs: List[Any], preview_chars: int = 220) -> List[Dict[str, Any]]:
    """
    rerank된 docs로 UI-friendly sources 생성.
    - evidence_no는 1..N으로 고정
    """
    sources: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs or [], 1):
        md = getattr(doc, "metadata", {}) or {}
        sources.append(
            {
                "evidence_no": i,
                "title": md.get("title", "관련 판례/자료"),
                "dept": md.get("dept", md.get("medical_dept", "진료과 없음")),
                "section": md.get("section", "section 없음"),
                "seq": md.get("seq"),
                "content_preview": _safe_preview(getattr(doc, "page_content", ""), preview_chars),
            }
        )
    return sources

_DOC_LIKE_PATTERNS = [
    r"^제목:\s*의료과실",
    r"\b신청인\b",
    r"\b피신청인\b",
    r"\b의료분쟁\s*조정신청서\b",
    r"\b손해배상\s*청구\b",
    r"\b증거\s*자료\b",
    r"\b요청\s*사항\b",
    r"\[작성일\]",
    r"문서 작성을 위해 아래 정보가 추가로 필요합니다",
]

def _is_doc_like_ai_message(text: str) -> bool:
    """
    Writer가 만든 '문서 본문'이 히스토리에 누적되면,
    다음 문서 요청에서 반복/증식 문제가 생김 → Writer 입력에서 제외.
    """
    t = (text or "").strip()
    if len(t) < 200:
        return False
    for p in _DOC_LIKE_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE | re.MULTILINE):
            return True
    # 너무 긴 텍스트는 문서일 확률 높음
    if len(t) > 2500:
        return True
    return False

def _history_to_text_for_writer(session_id: str, max_turns: int = 14) -> str:
    """
    Writer에 넣을 히스토리를 "상담 중심"으로 정리.
    - AI 문서 결과(DOC-like)는 제외
    - 최근 max_turns 턴만 사용
    """
    history = get_session_history(session_id)
    if not history.messages:
        return "이전 대화 기록 없음."

    # 최근 N개만
    msgs = history.messages[-max_turns:] if len(history.messages) > max_turns else history.messages

    lines: List[str] = []
    turn_idx = 0
    for msg in msgs:
        role = "의뢰인" if msg.type == "human" else "변호사"

        # ✅ AI 문서 결과는 제외 (반복/증식 방지)
        if msg.type != "human" and _is_doc_like_ai_message(msg.content):
            continue

        turn_idx += 1
        lines.append(f"### Turn {turn_idx} ({role})\n{msg.content}\n")

    return "\n".join(lines) if lines else "이전 대화 기록 없음."

def _ensure_session(session_id: str) -> None:
    _ = get_session_history(session_id)


# -------------------------------------------------------------------------
# [API] 통합 채팅 엔드포인트
#  - CHAT: answer_with_sources()만 사용 (중복검색/근거 불일치 제거)
#  - DOC: writer 입력에서 문서 결과 제거 + 세션 저장 안정화
# -------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Question):
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    session_id = _sanitize_session_id(request.session_id)
    query = _sanitize_query(request.query)

    print(f"\n📩 [{request_id}] Session={session_id} | Query={query}")

    # 세션 히스토리 항상 준비
    _ensure_session(session_id)

    # 1) Router
    try:
        t_router0 = time.perf_counter()
        intent = router_chain.invoke({"question": query}).strip().upper()
        t_router1 = time.perf_counter()
        print(f"🤖 [{request_id}] Router={intent} ({int((t_router1-t_router0)*1000)}ms)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Router 오류: {str(e)}")

    # -----------------------------------------------------------------
    # [Case A] DOC
    # -----------------------------------------------------------------
    if "DOC" in intent:
        print(f"📝 [{request_id}] 문서 작성 모드 진입")

        try:
            history_text = _history_to_text_for_writer(session_id=session_id, max_turns=14)

            # ✅ “🔴 현재 상태” 같은 반복 토큰을 매번 누적하지 않도록, 입력 구조를 고정
            full_context = (
                f"[상담 요약/대화 내역]\n{history_text}\n\n"
                f"[의뢰인의 현재 요청(최우선)]\n{query}\n"
            )

            t_doc0 = time.perf_counter()
            document_content = writing_chain.invoke({"chat_history": full_context})
            t_doc1 = time.perf_counter()

            # ✅ 메모리 저장: 항상 저장되도록
            hist = get_session_history(session_id)
            hist.add_message(HumanMessage(content=query))
            hist.add_message(AIMessage(content=document_content))

            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(
                f"✅ [{request_id}] DOC 생성 완료 "
                f"doc={int((t_doc1-t_doc0)*1000)}ms total={latency_ms}ms"
            )

            return {
                "request_id": request_id,
                "session_id": session_id,
                "type": "document",
                "answer": "요청하신 사항을 반영하여 문서를 작성했습니다. 아래 내용을 확인해주세요.",
                "document_content": document_content,
                "mode": None,
                "sources": [],
                "latency_ms": latency_ms,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DOC 처리 중 오류: {str(e)}")

    # -----------------------------------------------------------------
    # [Case B] CHAT (RAG)
    # -----------------------------------------------------------------
    print(f"💬 [{request_id}] 상담 모드 진입")

    try:
        # ✅ 단일 진실 소스: rag_pipeline에서 최종 mode/docs를 함께 반환
        t_rag0 = time.perf_counter()
        out = answer_with_sources(query, session_id=session_id)
        t_rag1 = time.perf_counter()

        answer = (out or {}).get("answer", "") or ""
        mode = (out or {}).get("mode")  # "SOLUTION"|"INTERVIEW"

        # 1) rag_pipeline이 docs를 주면 docs로 sources 생성
        docs = (out or {}).get("docs", []) or []
        sources = _build_sources_from_docs(docs)

        # 2) rag_pipeline이 sources(dict 리스트)를 직접 주는 구현이라면 그걸 우선 사용
        #    (필드가 더 풍부할 수 있음)
        if (out or {}).get("sources"):
            sources = (out or {}).get("sources")

        # ✅ 세션 전달 확실화: (rag_chain 내부에서도 session_id를 쓰는 경우 대비)
        # answer_with_sources에서 이미 처리했겠지만, 방어적으로 기록만 보장
        # (대부분 RunnableWithMessageHistory가 자동 기록)
        _ensure_session(session_id)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        print(
            f"✅ [{request_id}] RAG 완료 mode={mode} "
            f"rag={int((t_rag1-t_rag0)*1000)}ms total={latency_ms}ms sources={len(sources)}"
        )

        return {
            "request_id": request_id,
            "session_id": session_id,
            "type": "chat",
            "answer": answer,
            "document_content": None,
            "mode": mode,
            "sources": sources,
            "latency_ms": latency_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CHAT 처리 중 오류: {str(e)}")


# -------------------------------------------------------------------------
# [API] 추천 질문 (Chips)
# -------------------------------------------------------------------------
@app.get("/suggested_questions")
def get_suggestions():
    return {
        "questions": [
            "백내장 수술 부작용 판례 알려줘",
            "지금 상담 내용으로 내용증명서 써줘",
            "의료분쟁조정 신청 방법이 뭐야?",
            "설명 의무 위반이 인정된 사례 있어?",
        ]
    }


# -------------------------------------------------------------------------
# [API] 대화 내역 조회 (+ limit 지원)
# -------------------------------------------------------------------------
@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = Query(50, ge=1, le=200)):
    session_id = _sanitize_session_id(session_id)
    hist = get_session_history(session_id)

    messages = hist.messages[-limit:] if hist.messages else []
    return {
        "session_id": session_id,
        "count": len(messages),
        "history": [
            {"role": "user" if m.type == "human" else "ai", "content": m.content}
            for m in messages
        ],
    }


# -------------------------------------------------------------------------
# 실행:
#   uv run uvicorn main:app --reload
# -------------------------------------------------------------------------



