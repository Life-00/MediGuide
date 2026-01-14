import os, re, shutil
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ibm import WatsonxEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

load_dotenv()

IBM_URL = os.getenv("IBM_CLOUD_URL")
PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API = os.getenv("API_KEY")
PERSIST_DIR = "./chroma_db_fixed"
COLLECTION_NAME = "mediguide_cases"

def normalize_text(x: str) -> str:
    if x is None:
        return ""
    x = str(x).replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()

def ingest_data():
    print("📂 데이터 로딩 및 DB 구축 시작...")

    file_path = "test-data2.xlsx"
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    # 불필요 컬럼 제거
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # 텍스트 splitter (대략적인 길이 기준, 상황에 맞게 조절 가능)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,      # 임베딩 512 토큰 truncate를 고려해 넉넉히 쪼갬
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )

    docs = []
    print(f"🔹 총 {len(df)}개의 데이터를 처리합니다.")

    for _, row in df.iterrows():
        case_id = str(row.get("case_id", "unknown"))
        dept = str(row.get("medical_dept", "unknown"))
        title = str(row.get("title", "N/A"))
        seq = str(row.get("seq", ""))

        sections = {
            "overview": row.get("case_overview", ""),
            "issues": row.get("issues", ""),
            "solution": row.get("solution", ""),
            "result": row.get("result", ""),
            "final_result": row.get("final_result", ""),
        }

        for section_name, section_text in sections.items():
            section_text = normalize_text(section_text)
            if not section_text:
                continue

            # 섹션별 chunking
            chunks = splitter.split_text(section_text)
            for i, ch in enumerate(chunks):
                content = (
                    f"[사건명]: {title}\n"
                    f"[진료과목]: {dept}\n"
                    f"[섹션]: {section_name}\n\n"
                    f"{ch}"
                )

                metadata = {
                    "case_id": case_id,
                    "dept": dept,
                    "title": title,         # ✅ 근거 카드용
                    "seq": seq,             # ✅ 원문 링크 매핑용
                    "section": section_name,
                    "chunk_id": f"{case_id}:{section_name}:{i}",
                }

                docs.append(Document(page_content=content, metadata=metadata))

    # 임베딩 설정
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    embeddings = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url=IBM_URL,
        project_id=PROJECT_ID,
        params=embed_params,
        apikey=WATSONX_API,
    )

    # DB 재생성
    if os.path.exists(PERSIST_DIR):
        print("⚠️ 기존 DB 폴더가 존재합니다. 덮어쓰기를 진행합니다.")
        shutil.rmtree(PERSIST_DIR)

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ DB 구축 완료! docs={len(docs)} 저장 경로: {PERSIST_DIR}")

if __name__ == "__main__":
    ingest_data()
