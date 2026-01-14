# ingest.py
import os
import shutil
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings

load_dotenv()

# 설정
PERSIST_DIR = "./chroma_db_fixed"  # 기존 경로 그대로 쓰려면 유지
FILE_PATH = "test-data2.xlsx"

# Upstage 임베딩 모델 (필요하면 .env로 오버라이드 가능)
SOLAR_EMBED_MODEL = os.getenv("UPSTAGE_EMBED_MODEL", "solar-embedding-1-large")


def ingest_data():
    print("📂 데이터 로딩 및 DB 구축 시작...")

    # 1) 데이터 로드
    try:
        df = pd.read_excel(FILE_PATH)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {FILE_PATH}")
        return

    chunks = []
    print(f"🔹 총 {len(df)}개의 데이터를 처리합니다.")

    for idx, row in df.iterrows():
        content = f"""
[사건번호]: {row.get('case_id', 'N/A')}
[진료과목]: {row.get('medical_dept', 'N/A')}
[사건명]: {row.get('title', 'N/A')}
[사건개요]: {row.get('case_overview', 'N/A')}
[주요쟁점]: {row.get('issues', 'N/A')}
[해결/판결요지]: {row.get('solution', 'N/A')}
[처리결과]: {row.get('result', 'N/A')}
""".strip()

        metadata = {
            "id": str(idx),
            "case_id": str(row.get("case_id", "unknown")),
            "dept": str(row.get("medical_dept", "unknown")),
            "title": str(row.get("title", "관련 판례/자료")),
        }

        chunks.append(Document(page_content=content, metadata=metadata))

    # 2) 임베딩 설정 (Upstage)
    embeddings = UpstageEmbeddings(model=SOLAR_EMBED_MODEL)

    # 3) 기존 DB 삭제 후 재구축 (차원 불일치 방지)
    if os.path.exists(PERSIST_DIR):
        print("⚠️ 기존 DB 폴더가 존재합니다. 삭제 후 새로 구축합니다.")
        shutil.rmtree(PERSIST_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print(f"✅ DB 구축 완료! 저장 경로: {PERSIST_DIR}")


if __name__ == "__main__":
    ingest_data()
