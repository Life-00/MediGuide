# rag_chain.py
import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

load_dotenv()

# 전역 설정
IBM_URL = os.getenv('IBM_CLOUD_URL')
PROJECT_ID = os.getenv('PROJECT_ID')
WATSONX_API = os.getenv('API_KEY')
PERSIST_DIR = "./chroma_db_fixed"

def get_rag_chain():
    # 1. 임베딩 설정 (ingest.py와 동일해야 함)
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

    # 2. 저장된 DB 로드 (데이터 생성 X)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # 3. LLM 설정
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 1000,
            "min_new_tokens": 1,
            "temperature": 0.1,
            # [핵심 수정] AI에게 "여기서 멈춰!"라고 알려주는 신호입니다.
            # "질문:" 이라는 단어가 또 나오려고 하면 강제로 입을 다물게 합니다.
            "stop_sequences": ["\n질문:", "\n\n질문:", "질문:"]
        }
    )

    # 4. 프롬프트 & 체인
    template = """당신은 의료 분쟁 상담 AI입니다. 
    아래의 [참고 판례]를 바탕으로 사용자의 질문에 답변하세요.
    답변을 마친 후에는 절대로 새로운 질문을 생성하지 말고 종료하세요.
    
    [참고 판례]:
    {context}

    질문: {question}
    
    답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    print("🧪 RAG 로직 테스트 중...")
    
    
    
    
    try:
        chain = get_rag_chain()
        test_query = "백내장 수술 후 빛 번짐 부작용이 있어"
        
        print(f"\n❓ 질문: {test_query}")
        answer = chain.invoke(test_query)
        print(f"\n💡 답변:\n{answer}")
        print("\n✅ 로직 테스트 성공!")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")