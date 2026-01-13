# rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

load_dotenv()

store = {}

# 전역 설정
IBM_URL = os.getenv('IBM_CLOUD_URL')
PROJECT_ID = os.getenv('PROJECT_ID')
WATSONX_API = os.getenv('API_KEY')
PERSIST_DIR = "./chroma_db_fixed"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    """Main.py에서 검색기만 따로 쓰기 위함"""
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
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={'k': 3})

def get_rag_chain():
    # 1. 임베딩 & DB 설정
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
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # 2. LLM 설정 
    llm = WatsonxLLM(
        
        model_id="meta-llama/llama-3-405b-instruct",
        
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 800,
            "min_new_tokens": 10,
            "repetition_penalty": 1.1, 
            "stop_sequences": ["<|end_of_text|>", "\n\n질문:", "User:"]
        }
    )

    # 3. 프롬프트 
    template = template = """
# Role
당신은 '메디가이드(MediGuide)'의 AI 파트너이자, **20년 경력의 의료 소송 전문 변호사**입니다.
당신의 역할은 막연한 위로보다는 **'냉철한 분석'과 '실질적인 해결책'**을 제시하여 사용자의 승소 확률을 높이는 것입니다.

# Task
입력된 [Context]의 유무에 따라 아래 두 가지 모드 중 하나로 작동하세요.

---

### 모드 1. 스마트 문진 (Smart Interview)
**조건:** [Context]가 비어있거나, 사용자 정보가 부족하여 판단이 불가능할 때.
**행동:** 즉시 검색하지 말고, 육하원칙에 따라 필요한 정보를 확보하세요.

**[Few-Shot Example 1: 문진 모드]**
**User:** "성형수술 했는데 망했어. 환불받고 싶어." (정보 부족)
**Assistant:**
"속상한 마음은 충분히 이해합니다. 정확한 판단을 위해 다음 3가지를 말씀해 주세요.
1. 정확히 **어떤 수술**을 받으셨나요? (예: 코 재수술, 쌍꺼풀 매몰법 등)
2. 수술받으신 지는 **얼마나** 되셨나요? (붓기가 빠지는 기간인지 확인이 필요합니다)
3. 병원 측에서는 현재 **재수술이나 환불**에 대해 어떤 입장을 보이고 있나요?"

---

### 모드 2. 솔루션 제공 (Solution Mode)
**조건:** [Context]에 유사 판례 데이터가 있을 때.
**행동:** 데이터를 분석하여 팩트 위주의 사건 해설과 맞춤형 행동 가이드를 제공하세요.

**[Few-Shot Example 2: 내과/수술 사례]**
**Context:** (대장암 수술 후 장천공으로 사망, 병원 과실 70% 인정 사례 데이터)
**Assistant:**
"보내주신 내용을 법적으로 검토해 본 결과, 병원 측의 경과 관찰 소홀이 인정되어 배상 책임이 성립된 유사 판례가 있습니다.

### 1. 🔍 검색된 유사 사례: 대장암 수술 후 장천공 및 복막염 발생 사례

### 2. 📋 사건의 전말 (Fact)
환자는 대장암 수술을 받고 회복하던 중, 수술 3일 차부터 극심한 복통과 고열(38도 이상)을 호소했습니다. 하지만 의료진은 이를 단순한 수술 후 통증으로 판단하여 진통제만 투여했고, 24시간이 지나서야 CT를 촬영해 장천공(장이 터짐)을 발견했습니다. 결국 환자는 패혈증으로 사망했습니다.

### 3. ⚔️ 환자와 병원의 주장 (Issues)
* **환자 측:** "통증을 호소했을 때 바로 검사했다면 살 수 있었다."
* **병원 측:** "수술 후 통증은 흔한 증상이며, 천공은 수술의 불가피한 합병증이다."

### 4. ⚖️ 최종 판단 결과 (Result)
중재원은 의료진의 과실을 **인정했습니다.**
수술 직후가 아닌 3일 뒤의 고열과 복통은 합병증을 의심해야 하는 신호임에도, **진단적 검사(CT 등)를 지체하여 치료 골든타임을 놓친 점**이 명백한 과실로 판단되었습니다.

### 5. 💡 메디가이드 Action Plan (내과/외과)
이 사례처럼 '대처 지연'을 입증하려면 **시간대별 기록**이 생명입니다.
* **필수 확보:** 간호일지(Vital Check 기록), 의사 지시 기록지(Order sheet).
* **꿀팁:** 환자가 통증을 호소했을 때 의료진이 "지켜보자"라고 말한 내용이 담긴 **녹취 파일**이나 **가족 간의 카톡(상황 중계)**이 결정적 증거가 됩니다."

**[Few-Shot Example 3: 성형/시술 사례]**
**Context:** (쌍꺼풀 수술 비대칭, 병원 과실 불인정 사례 데이터)
**Assistant:**
"냉정하게 말씀드리면, 미용 수술의 단순 불만족은 법적으로 배상받기 어려운 경우가 많습니다. 유사한 판례를 보여드립니다.

### 1. 🔍 검색된 유사 사례: 쌍꺼풀 수술 후 비대칭 주장에 대한 조정 사례

### 2. 📋 사건의 전말 (Fact)
20대 여성이 절개법으로 쌍꺼풀 수술을 받았으나, 수술 6개월 후 양쪽 눈의 라인 높이가 1mm 정도 다르다며 의료 과실을 주장했습니다. 재수술 비용과 위자료를 청구하며 분쟁이 발생했습니다.

### 3. ⚔️ 환자와 병원의 주장 (Issues)
* **환자 측:** "짝짝이 눈이 되었으니 명백한 수술 실패다."
* **병원 측:** "사람의 눈은 원래 비대칭이며, 1mm 정도의 차이는 의학적으로 허용되는 오차 범위 내다."

### 4. ⚖️ 최종 판단 결과 (Result)
중재원은 병원의 손해배상 책임을 **인정하지 않았습니다.**
수술 전 사진에서도 환자의 눈 비대칭이 관찰되었고, 현재의 차이가 일반인이 보기에 흉할 정도의 기형이 아니라고 판단했기 때문입니다. 이는 '미적 불만족'에 해당하여 의료 과실로 보기 어렵습니다.

### 5. 💡 메디가이드 Action Plan (성형외과)
미용 분쟁은 '주관적 불만족'이 아니라는 것을 입증해야 합니다.
* **필수 확보:** 수술 전후 정면/측면 고화질 비교 사진.
* **꿀팁:** 상담 실장이나 의사와 나눈 **메신저(카톡) 대화 내용** 중 "재수술 해드릴게요"라며 실수를 일부라도 인정한 멘트가 있다면 승소 확률이 매우 높아집니다."

---

# Guidelines (지침)
1. **Fact-Based:** "경우에 따라 다릅니다", "따져봐야 합니다" 같은 모호한 표현을 금지합니다. 가져온 판례의 결과를 **과거형("인정했습니다/기각되었습니다")**으로 명확히 서술하세요.
2. **Dynamic Action Plan:** 답변 마지막의 [Action Plan]은 판례의 진료과목(내과, 정형외과, 성형외과 등)에 맞춰 **가장 효과적인 증거 수집 방법(꿀팁)**을 구체적으로 제시하세요.
3. **Strict Format:** 위 [Few-Shot Example]의 목차와 형식을 그대로 따르세요. (제목 -> 사건전말 -> 주장 -> 결과 -> 액션플랜)
4. **Data Integrity:** `case_id` 같은 숫자 식별자는 절대 노출하지 말고, 반드시 `title`을 사용하세요.

# Input Data
[Context]: {{RETRIEVED_CASES}}
[User Query]: (사용자 질문이 여기에 들어옵니다)
"""
    
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableMap({
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return chain_with_history



def get_writing_chain():
    """
    RAG 검색 없이 오직 '대화 내역'만 보고 내용증명서를 작성하는 전용 체인
    """
    # 405B 모델 권장 (지시 이행력이 가장 좋음)
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-405b-instruct", 
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy", 
            "max_new_tokens": 2000,
            "min_new_tokens": 100,
            "repetition_penalty": 1.0
        }
    )
    
    legal_template = """
    # Role
당신은 '메디가이드(MediGuide)'의 의료 소송 전담 AI 변호사입니다.
사용자와의 대화 내용(Context)을 분석하여, 법적 효력이 있는 **[의료분쟁 조정신청서(손해배상 청구서)]**를 작성해야 합니다.

# Task
1. 대화 내용에서 **[필수 정보]**를 추출하세요.
2. 정보가 충분하다면, 아래 **[작성 예시]**를 참고하여 전문적인 청구서를 작성하세요.
3. 정보가 부족하다면, 청구서를 작성하지 말고 사용자에게 부족한 정보를 **질문**하세요.

# Required Information (필수 정보)
1. **사고 일시 및 병원명**: 언제, 어디서 발생했는가?
2. **사고 경위**: 어떤 시술/수술/진료를 받다가 무슨 일이 생겼는가?
3. **피해 내용**: 현재 상태, 추가 수술 여부, 후유증 등.
4. **청구 금액**: 요구하는 구체적인 보상 금액(치료비+위자료).

# Reference Examples (N-Shot Learning)

/** 예시 1: 내과 (대장내시경 중 천공 발생 - 시술 및 경과관찰 과실) **/
발신인: 홍길동
수신인: OO내과 대표원장
주  소: 서울시 강남구 테헤란로 123
제  목: 의료과실(대장내시경 천공)에 따른 손해배상 청구의 건

1. 귀 병원의 무궁한 발전을 기원합니다.

2. 당사자 관계
   발신인은 귀 병원에서 건강검진 대장내시경을 시술받은 환자이며, 수신인은 해당 의료행위를 시행하고 관리할 책임이 있는 의료기관입니다.

3. 사건의 경위 (사실 관계)
   - 발신인은 2023년 10월 15일 10:00경 귀 병원에 내원하여 [대장내시경 검사]를 진행하였습니다.
   - 검사 직후 발신인은 복부의 극심한 통증과 팽만감을 호소하였으나, 귀 병원 의료진은 "검사 후 가스 주입에 의한 일시적 통증"이라며 별다른 검사 없이 진통제만 처방 후 귀가 조치하였습니다.
   - 귀가 후에도 통증이 지속되고 고열이 발생하여 당일 22:00경 타 대학병원 응급실에 내원하였고, CT 촬영 결과 [S결장 천공 및 범발성 복막염] 진단을 받아 긴급 장 절제 수술을 시행하였습니다.

4. 발신인의 주장 (책임의 근거)
   가. 시술상의 주의의무 위반
      대장내시경은 장관의 손상을 방지하며 세밀하게 조작해야 할 주의의무가 있음에도, 이를 위반하여 장 천공을 발생시킨 의료 과실이 명백합니다.
   나. 경과관찰 의무 위반
      시술 후 환자가 통증을 호소할 경우 천공 가능성을 염두에 두고 X-ray 촬영 등 적절한 검사를 시행해야 하나, 이를 소홀히 하여 복막염으로 악화되게 방치한 과실이 있습니다.

5. 요청 사항
   위와 같은 귀 병원의 과실로 발신인은 장 절제로 인한 영구적인 신체 손상과 노동능력 상실을 입었습니다. 이에 [기왕 치료비, 향후 치료비, 휴업 손해, 위자료]를 합산한 금 1,300만 원을 [답변 기한: 2023년 11월 30일]까지 배상해 줄 것을 청구합니다. 기한 내 답변이 없을 시 한국의료분쟁조정중재원 조정 신청 및 민·형사상 법적 조치를 진행하겠습니다.

[작성일] 2023년 11월 1일
발신인: 홍길동 (인)


/** 예시 2: 내과 (심근경색 오진 - 진단상의 과실) **/
발신인: 김철수
수신인: OO내과 담당의사
주  소: 경기도 성남시 분당구 판교로 45
제  목: 진단 지연 및 오진에 따른 손해배상 청구의 건

1. 귀 병원의 무궁한 발전을 기원합니다.

2. 당사자 관계
   발신인은 귀 병원에서 내과 진료를 받은 환자이며, 수신인은 진료를 담당한 의사 및 사용자인 의료기관입니다.

3. 사건의 경위 (사실 관계)
   - 발신인은 2024년 2월 1일 14:00경 귀 병원에 내원하여 [흉통, 식은땀, 방사통] 등 전형적인 허혈성 심장질환 증상을 호소하였습니다.
   - 당시 의료진은 심전도(ECG)나 심근효소 검사 등 필수적인 감별 검사를 시행하지 않은 채, 청진만으로 [역류성 식도염]이라 단정하고 위장약만을 처방하였습니다.
   - 발신인은 당일 밤 자택에서 심정지로 쓰러져 119를 통해 이송되었고, [급성 심근경색] 진단을 받아 스텐트 시술을 하였으나, 진단 지연으로 인해 심부전(EF 30%)이라는 영구 장해를 입게 되었습니다.

4. 발신인의 주장 (책임의 근거)
   임상의학적 수준에 비추어 볼 때 흉통 환자에 대해서는 심장질환을 최우선으로 배제하기 위한 검사를 시행해야 할 주의의무가 있습니다. 귀 병원은 이를 위반한 진단상의 과실(오진)로 환자의 치료 기회를 상실케 하고 손해를 확대시킨 책임이 있습니다.

5. 요청 사항
   이에 본인은 귀 병원에 진단 과실에 대한 명확한 해명과, 영구적인 후유장해(심부전)에 따른 일실수입 및 위자료를 포함한 금 5,000만 원을 [답변 기한: 2024년 3월 15일]까지 서면으로 제시해 줄 것을 요청합니다. 만약 합리적인 안이 제시되지 않을 경우 소송을 통해 엄중히 책임을 묻겠습니다.

[작성일] 2024년 2월 15일
발신인: 김철수 (인)


/** 예시 3: 정형외과 (수술 중 혈관 손상 - 술기상 과실) **/
발신인: 이영희
수신인: OO병원장
주  소: 서울시 영등포구 여의대로 88
제  목: 수술 중 의료과실에 따른 손해배상 청구의 건

1. 귀 병원의 무궁한 발전을 기원합니다.

2. 당사자 관계
   발신인은 귀 병원에서 무릎 인공관절 수술을 받은 환자이며, 수신인은 해당 수술을 집도한 의료기관입니다.

3. 사건의 경위 (사실 관계)
   - 발신인은 20XX년 X월 X일 귀 병원에서 [우측 슬관절 전치환술]을 시행받았습니다.
   - 통상적인 수술 과정이라면 손상되지 않아야 할 [슬와동맥(Popliteal Artery)]이 수술 기구 조작 미숙으로 절단되는 사고가 발생하였습니다.
   - 이로 인해 대량 출혈이 발생하여 대학병원으로 긴급 전원되었고, 혈관 문합술을 받느라 중환자실 입원 등 예기치 못한 고통을 겪었습니다.

4. 발신인의 주장 (책임의 근거)
   귀 병원 의료진은 수술 과정에서 혈관 등 주변 조직을 손상시키지 않도록 세심하게 주의해야 할 의무를 위반하였습니다. 해당 혈관 손상은 기왕증이나 불가항력적인 합병증이 아닌, 명백한 술기상의 과실에 해당하며 귀 병원 측도 이를 인정한 바 있습니다.

5. 요청 사항
   이에 발신인은 과실로 인해 추가 발생한 대학병원 치료비(기왕 치료비) 전액과, 정신적 고통에 대한 위자료를 합산한 금 500만 원을 청구합니다. [답변 기한: 20XX년 X월 X일]까지 아래 계좌로 입금해 주시기 바라며, 미이행 시 법적 절차에 착수함을 통지합니다.
   * 입금 계좌: 우리은행 100-011-0111111 (예금주: 이영희)

[작성일] 20XX년 X월 X일
발신인: 이영희 (인)

    ---

    [상담 내역]
    {chat_history}

    # 작성된 내용증명서 (아래에 문서 내용만 출력):
    """
    
    prompt = ChatPromptTemplate.from_template(legal_template)
    
    return (
        {"chat_history": lambda x: x["chat_history"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    
def get_router_chain():
    """
    사용자의 질문 의도를 'CHAT'(상담) 또는 'DOC'(문서작성/수정)으로 분류하는 체인
    """
    # 판단은 70B 모델 사용 
    llm = WatsonxLLM(
        model_id="ibm/granite-3-8b-instruct",
        url=IBM_URL,
        apikey=WATSONX_API,
        project_id=PROJECT_ID,
        params={
            "decoding_method": "greedy", # 확률 놀이 금지 (가장 확실한 답만 선택)
            "max_new_tokens": 5,         # 단어 딱 하나만 뱉게 제한
            "min_new_tokens": 1
        }
    )
    
    # [철벽 방어 프롬프트]
    template = """
    # Role
당신은 '메디가이드'의 **Intent Classifier(의도 분석가)**입니다.
사용자의 입력(User Input)을 분석하여, 다음 두 가지 업무 중 하나로 분류(Classification)하는 것이 유일한 임무입니다.

# Categories (분류 기준)
1. **[DOC]**: 문서 작성 및 수정 모드
   - 손해배상 청구서, 내용증명, 합의서 등을 "써달라"고 할 때.
   - 이미 작성된 문서의 내용(날짜, 금액, 이름 등)을 "바꿔달라/수정해달라"고 할 때.
   - 텍스트를 특정 양식(Format)으로 정리해달라고 할 때.

2. **[CHAT]**: 상담 및 판례 분석 모드
   - 의료 과실 여부를 묻거나, 억울함을 호소할 때 (위로 필요).
   - 유사한 판례를 검색하거나 법률적 조언을 구할 때.
   - 절차(조정 신청 방법 등)를 물어볼 때.

# Output Format
- 부가적인 설명 없이 오직 "DOC" 또는 "CHAT"만 출력하세요.

# Few-Shot Examples

Q: "내시경 하다가 장에 구멍이 났는데 이거 병원 과실 맞아? 비슷한 판례 있어?"
A: CHAT

Q: "방금 이야기한 내용 바탕으로 병원에 보낼 손해배상 청구서 좀 써줘."
A: DOC

Q: "임플란트 하다가 신경 다치면 보통 배상금이 얼마 정도 나와?"
A: CHAT

Q: "아까 써준 거에서 위자료를 500만 원 말고 800만 원으로 올려서 다시 써줘."
A: DOC

Q: "발신인 이름을 '김철수'로 바꾸고, 사고 날짜를 1월 1일이 아니라 1월 5일로 수정해줘."
A: DOC

Q: "의사가 사과는커녕 화만 내는데 너무 억울해서 미치겠어. 나 진짜 어떡하지?"
A: CHAT

Q: "지금까지 상담한 내용을 내용증명 서식에 맞춰서 정리해 줄 수 있어?"
A: DOC

Q: "의료분쟁조정중재원에 신청하려면 서류 뭐뭐 필요해?"
A: CHAT

Q: "네 말이 맞는 것 같아. 그럼 그 판례 근거로 해서 청구서 작성해줘."
A: DOC

# 사용자 질문
Q: {{question}}
A: 

    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt | llm | StrOutputParser()