# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio

# get_retriever 함수 추가로 가져오기
try:
    from rag_pipeline import get_rag_chain, get_retriever, get_session_history
except ImportError:
    from src.mediguide_rag.rag_pipeline import get_rag_chain, get_retriever, get_session_history

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 체인 및 검색기 로딩
chain = get_rag_chain()
retriever = get_retriever() 

class Question(BaseModel):
    query: str
    session_id: str = "default_user"

# [기능 1] 일반 채팅 (기존)
@app.post("/chat")
async def chat_endpoint(request: Question):
    answer = chain.invoke(
        {"question": request.query}, 
        config={"configurable": {"session_id": request.session_id}}
    )
    return {"answer": answer}

# [기능 2] 답변 스트리밍 (타자 치는 효과) 
@app.post("/chat/stream")
async def chat_stream_endpoint(request: Question):
    async def generate():
        # 체인에서 한 글자씩 받아옴
        async for chunk in chain.astream(
            {"question": request.query}, 
            config={"configurable": {"session_id": request.session_id}}
        ):
            # chunk가 문자열이면 바로 전송
            if chunk:
                yield chunk 

    return StreamingResponse(generate(), media_type="text/event-stream")

# [기능 3] 답변 + 근거 자료(출처) 같이 주기 
@app.post("/chat_with_sources")
async def chat_with_sources(request: Question):
    # 1. 먼저 검색을 수행해서 관련 판례(Docs)를 가져옴
    docs = retriever.invoke(request.query)
    
    # 2. 문서에서 메타데이터(제목, 사건번호 등) 추출하여 리스트로 만듦
    sources = []
    for doc in docs:
        sources.append({
            "title": doc.metadata.get("title", "제목 없음"),
            "case_id": doc.metadata.get("case_id", "번호 없음"),
            "dept": doc.metadata.get("dept", "진료과 없음"),
            "content_preview": doc.page_content[:150] + "..."
        })

    # 3. 답변 생성
    answer = chain.invoke(
        {"question": request.query}, 
        config={"configurable": {"session_id": request.session_id}}
    )
    
    # 4. 답변과 출처를 묶어서 반환
    return {
        "answer": answer,
        "sources": sources 
    }

# [추가] 대화 내역 조회 (새로고침 대응용)
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    # rag_pipeline.py에 get_history_list 함수가 없다면 이 부분은 에러날 수 있음
    # 그럴 땐 아래 return {"history": []} 로 임시 처리
    try:
        from rag_pipeline import store
        if session_id in store:
            messages = store[session_id].messages
            return {"history": [{"role": "user" if m.type=="human" else "ai", "content": m.content} for m in messages]}
        return {"history": []}
    except:
        return {"history": []}
# 실행: uv run uvicorn main:app --reload
