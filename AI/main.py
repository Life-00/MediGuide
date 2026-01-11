# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.mediguide_rag.rag_pipeline import get_rag_chain


app = FastAPI()

# 서버 시작 시 체인 로딩 (한 번만 로딩하면 됨)
chain = get_rag_chain()

class Question(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: Question):
    # 미리 로딩된 체인 사용 -> 속도 빠름
    answer = chain.invoke(request.query)
    return {"answer": answer}

# 실행: uv run uvicorn main:app --reload
