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
    file_path = "test-data2.xlsx" 
    
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    chunks = []
    print(f"🔹 총 {len(df)}개의 데이터를 처리합니다.")

    for idx, row in df.iterrows():
        # [핵심 수정: 새 데이터셋 컬럼명에 맞게 매핑]
        # 새 파일의 헤더: case_id, medical_dept, title, case_overview, issues, solution, result
        content = f"""
        [사건번호]: {row.get('case_id', 'N/A')}
        [진료과목]: {row.get('medical_dept', 'N/A')}
        [사건명]: {row.get('title', 'N/A')}
        [사건개요]: {row.get('case_overview', 'N/A')}
        [주요쟁점]: {row.get('issues', 'N/A')}
        [해결/판결요지]: {row.get('solution', 'N/A')}
        [처리결과]: {row.get('result', 'N/A')}
        """
        
        # 메타데이터도 새 컬럼명에 맞게 수정
        metadata = {
            "id": str(idx), 
            "case_id": str(row.get('case_id', 'unknown')),
            "dept": str(row.get('medical_dept', 'unknown')) # 필터링용으로 과목 추가 추천
        }
        
        chunks.append(Document(page_content=content.strip(), metadata=metadata))

    # 2. 임베딩 설정
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
        print("⚠️ 기존 DB 폴더가 존재합니다. 덮어쓰기를 진행합니다.")
        # 안전하게 새로 만들고 싶다면 아래 주석 해제 (기존 DB 삭제)
        import shutil
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"✅ DB 구축 완료! 저장 경로: {PERSIST_DIR}")

if __name__ == "__main__":
    ingest_data()