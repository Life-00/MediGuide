# ingest.py
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ibm import WatsonxEmbeddings
from langchain_chroma import Chroma
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

load_dotenv()

# 설정
IBM_URL = os.getenv('IBM_CLOUD_URL')
PROJECT_ID = os.getenv('PROJECT_ID')
WATSONX_API = os.getenv('API_KEY')
PERSIST_DIR = "./chroma_db_fixed"  # DB 저장 경로

def ingest_data():
    print("📂 데이터 로딩 및 DB 구축 시작...")
    
    # 1. 데이터 로드
    df = pd.read_excel("test-data.xlsx")
    chunks = []
    for idx, row in df.iterrows():
        content = f"""
        [사건번호]: {row.get('Case', 'N/A')}
        [진료과목]: {row.get('진료과목 (medical_dept)', 'N/A')}
        [수술명]: {row.get('시술/수술명 (procedure_name)', 'N/A')}
        [부작용]: {row.get('부작용/증상 (symptom)', 'N/A')}
        [쟁점]: {row.get('주요 쟁점 (legal_issues)', 'N/A')}
        [결과]: {row.get('조정 결과 (result)', 'N/A')}
        [판례원문]: {row.get('판례 원문 (original_text)', 'N/A')}
        """
        metadata = {"id": str(idx), "case_id": str(row.get('Case', 'unknown'))}
        chunks.append(Document(page_content=content.strip(), metadata=metadata))

    # 2. 임베딩 설정 (DB 만들 때와 읽을 때 똑같아야 함!)
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

    # 3. 벡터 DB 생성 및 저장
    if os.path.exists(PERSIST_DIR):
        print("⚠️ 기존 DB가 존재합니다. 삭제하거나 덮어씁니다.")
        # shutil.rmtree(PERSIST_DIR) # 필요하면 기존 폴더 삭제 코드 추가
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"✅ DB 구축 완료! 저장 경로: {PERSIST_DIR}")

if __name__ == "__main__":
    ingest_data()