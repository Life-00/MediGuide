import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage

# -------------------------------------------------------------------------
# [Import] rag_pipeline.py에서 필요한 체인과 메모리 저장소 가져오기
# -------------------------------------------------------------------------
try:
    from rag_pipeline import (
        get_rag_chain,      # 상담용 (RAG)
        get_writing_chain,  # 문서 작성용 (Writer)
        get_router_chain,   # 의도 분류용 (Router)
        get_retriever,      # 근거 자료 검색용
        store               # 대화 내역 저장소 (메모리)
    )
except ImportError:
    # 폴더 구조 예외 처리
    from src.mediguide_rag.rag_pipeline import (
        get_rag_chain, get_writing_chain, get_router_chain, get_retriever, store
    )

# -------------------------------------------------------------------------
# [Setup] FastAPI 앱 초기화
# -------------------------------------------------------------------------
app = FastAPI(title="MediGuide AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
retriever = get_retriever()
print("✅ 로딩 완료! 서버가 준비되었습니다.")

# 요청 데이터 모델
class Question(BaseModel):
    query: str
    session_id: str = "default_user"

# -------------------------------------------------------------------------
# [API] 통합 채팅 엔드포인트 (Smart Router + Fix 적용)
# -------------------------------------------------------------------------
@app.post("/chat")
async def chat_endpoint(request: Question):
    print(f"\n📩 [요청 수신] Session: {request.session_id} | Query: {request.query}")
    
    try:
        # 1. 의도 파악 (Router AI 호출)
        intent = router_chain.invoke({"question": request.query}).strip().upper()
        print(f"🤖 [Router 판단] 의도: {intent}")

        # -----------------------------------------------------------------
        # [Case A] 문서 작성 요청 (DOC)
        # 예: "내용증명서 써줘", "500만원으로 수정해줘"
        # -----------------------------------------------------------------
        if "DOC" in intent:
            print("📝 [Process] 문서 작성 모드 진입")
            
            # (1) 과거 대화 기록 가져오기
            history_text = ""
            if request.session_id in store:
                for msg in store[request.session_id].messages:
                    role = "의뢰인" if msg.type == "human" else "변호사"
                    history_text += f"- {role}: {msg.content}\n"
            else:
                history_text = "이전 대화 기록 없음."

            # 🚨 [핵심 수정] 과거 기록 + '현재 요청사항'을 합쳐서 전달해야 함
            # 이걸 안 하면 AI가 "수정해줘"라는 말을 못 듣고 옛날 문서만 또 씀
            full_context = history_text + f"\n🔴 [의뢰인의 현재 요청사항(가장 중요)]: {request.query}\n"

            # (2) Writer LLM 호출
            document_content = writing_chain.invoke({"chat_history": full_context})

            # (3) [Memory Sync] 수동으로 기억 저장
            # RAG 체인과 달리, 여기서 직접 store에 넣어줘야 대화가 끊기지 않음
            if request.session_id in store:
                store[request.session_id].add_message(HumanMessage(content=request.query))
                store[request.session_id].add_message(AIMessage(content=document_content))

            # (4) 응답 반환
            return {
                "answer": "요청하신 사항을 반영하여 문서를 작성했습니다. 아래 내용을 확인해주세요.",
                "type": "document",
                "document_content": document_content,
                "sources": []
            }

        # -----------------------------------------------------------------
        # [Case B] 일반 법률 상담 (CHAT)
        # 예: "의료사고인가요?", "판례 알려줘"
        # -----------------------------------------------------------------
        else:
            print("💬 [Process] 일반 상담 모드 진입")

            # (1) 근거 자료 검색
            docs = retriever.invoke(request.query)
            sources = []
            for doc in docs:
                sources.append({
                    "title": doc.metadata.get("title", "관련 판례/자료"),
                    "case_id": doc.metadata.get("case_id", "정보 없음"),
                    "content_preview": doc.page_content[:200] + "..." # 미리보기 길이 늘림
                })

            # (2) RAG 답변 생성
            # RunnableWithMessageHistory가 자동으로 store 업데이트 함
            answer = rag_chain.invoke(
                {"question": request.query}, 
                config={"configurable": {"session_id": request.session_id}}
            )

            # (3) 응답 반환
            return {
                "answer": answer,
                "type": "chat",
                "document_content": None,
                "sources": sources
            }

    except Exception as e:
        print(f"❌ 에러 발생: {str(e)}")
        # 에러 나도 서버 안 죽고 프론트에 알려주기
        return {
            "answer": "죄송합니다. 일시적인 오류가 발생했습니다.", 
            "type": "error", 
            "error": str(e)
        }

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
            "설명 의무 위반이 인정된 사례 있어?"
        ]
    }

# -------------------------------------------------------------------------
# [API] 대화 내역 조회
# -------------------------------------------------------------------------
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id in store:
        messages = store[session_id].messages
        return {
            "history": [
                {
                    "role": "user" if m.type == "human" else "ai", 
                    "content": m.content
                } for m in messages
            ]
        }
    return {"history": []}
# 실행: uv run uvicorn main:app --reload
