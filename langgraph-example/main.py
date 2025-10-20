import logging
import os
from dotenv import load_dotenv
from typing import TypedDict, Dict, Optional, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
import math, ast, re, json, datetime
import pandas as pd
from bs4 import BeautifulSoup
import requests
from rag.rag_cancer_stats import get_cancer_survival_rate

# ======================
# 로거 설정
# ======================
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("app")

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY and os.path.exists("api_key.txt"):
    with open("api_key.txt", "r", encoding="utf-8") as f:
        OPENAI_API_KEY = f.read().strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 api_key.txt를 확인하세요.")

# =============================
# 2) 데이터 로드 (계산용 CSV)
# =============================
BASEHAZ_PATH = "data/Anujin_240828_ACS_baseline.csv"
COEFS_PATH   = "data/coefficients.csv"
try:
    basehazard_acs = pd.read_csv(BASEHAZ_PATH)
    coefs_acs      = pd.read_csv(COEFS_PATH)
except Exception as e:
    raise RuntimeError(f"데이터 로드 실패: {e}")

# ======================
# 1. 상태 정의 (TypedDict)
# ======================
class InputState(TypedDict):
    raw_query: str   # 사용자가 입력한 자연어 질문
    query_type: Optional[str]  # 질문 유형 (예: '생존분석', '일반질문' 등)
    extracted: Dict[str, Any]  # 변수 추출 결과

class ModelSelectState(TypedDict):
    raw_query: str   # 나이, 성별, stage, 기간 등 파싱된 구조화 데이터
    model_id: str
    extracted: Dict[str, Any]        # 선택된 모델 이름

class CalcState(TypedDict):
    survival_prob: float
    extracted: Dict[str, Any]

class OutputState(TypedDict):
    answer: str      # 최종 사용자 응답 메시지


# ======================
# 2. 노드 함수 정의
# ======================

classifier = ChatOpenAI(model="gpt-4.1", temperature=0)

def classify_question(state: InputState) -> InputState:
    query = state["raw_query"]

    prompt = f"""
    너는 입력 질문을 2가지 중 하나로 분류하는 역할이야.
    반드시 아래 중 하나만 출력해:
    - survival : 생존분석/생존 확률 계산 관련
    - chat     : 일반 의학/일반 대화

    질문: "{query}"
    답변:
    """
    ans = classifier.invoke(prompt).content.strip().lower()

    if ans not in ["survival", "chat"]:
        ans = "chat"  # fallback

    LOGGER.info(f"[NODE] classify_question | raw_query={query}, query_type={ans}")
    return {**state, "query_type": ans}

## Model에서 Variable 추출

extraction_llm = ChatOpenAI(model="gpt-4.1", temperature=0)

def variable_extraction(state: InputState) -> InputState:
    """사용자 입력(raw_query)에서 age, sex, stage, year를 추출해 JSON 반환"""
    query = state["raw_query"]
    prompt = f"""
    User question: "{query}"

    Task: Extract survival fields from the text below.
    Respond strictly in JSON format with keys: age, sex, stage, year.
    
    - age: integer
    - sex: male=0, female=1
    - stage: 1=Localized, 2=Regional, 3=Distant, 4=Unknown
    - year: survival duration in years (not calendar year)
    """

    # LLM 호출
    messages = [HumanMessage(content=prompt)]
    response = extraction_llm.invoke(messages)

    # JSON 파싱 (에러 시 기본값 반환)
    try:
        parsed = json.loads(response.content)
    except Exception:
        parsed = {"age": None, "sex": None, "stage": None, "year": None}

    # state 업데이트
    state["extracted"] = parsed

    LOGGER.info(f"[NODE] variable_extraction | extracted_variablees={parsed}")

    return state

## Retreiver 먼저 정의

MODEL_DOCS = [
    {"name": "model_1", "desc": "Uses only sex variable"},
    {"name": "model_2", "desc": "Uses only age variable"},
    {"name": "model_3", "desc": "Uses only seer stage variable"},
    {"name": "model_4", "desc": "Uses sex and age variables"},
    {"name": "model_5", "desc": "Uses sex and seer stage variables"},
    {"name": "model_6", "desc": "Uses age and seer stage variables"},
    {"name": "model_7", "desc": "Uses sex, age, and seer stage variables"},
]
texts = [f"{d['name']}: {d['desc']}" for d in MODEL_DOCS]

embeddings = OpenAIEmbeddings()
model_db = FAISS.from_texts(texts, embeddings)
model_retriever = model_db.as_retriever()

model_select_llm = ChatOpenAI(model="gpt-4.1", temperature=0)

def select_model(state: InputState) -> ModelSelectState:
    extracted = state.get("extracted", {})
    present_vars = [k for k, v in extracted.items() if v is not None]

    # Retriever로 후보군 검색
    enriched_query = f"Variables detected: {', '.join(present_vars)}"
    candidates = model_retriever.invoke(enriched_query)
    top_k = 5  # or 3
    candidates = candidates[:top_k]
    candidate_texts = "\n".join([doc.page_content for doc in candidates])

    # LLM이 후보 중 최적 모델 선택
    prompt = f"""
    Below are available survival models and their descriptions:

    {candidate_texts}

    The user's variables are: {', '.join(present_vars)}

    Task: Select the best-fitting model among the candidates above.
    Output ONLY the model name (e.g. "model_4").
    """

    response = model_select_llm.invoke(prompt).content.strip()

    LOGGER.info(f"[NODE] select_model | extracted={extracted}, result={response}")
    state["model_id"] = response
    return state

# def parse_input(state: ModelSelectState) -> ModelSelectState:
#     query = state["raw_query"]
#     parsed = {"age": 60, "sex": "F", "stage": 2, "horizon": 5}
#     LOGGER.info(f"[NODE] parse_input | query={query} -> parsed={parsed}")
#     return {"parsed_input": parsed, "model_id": ""}


def run_calculation(state: ModelSelectState) -> CalcState:
    """내부 모델을 통한 생존확률 계산 함수 (model_7)."""
    model_id = state["model_id"]

    if model_id != "model_7":
        LOGGER.warning(f"[NODE] run_calculation | model_id={model_id} not supported.")
        return {"survival_prob": None}
    
    parsed = state["extracted"]
    age = parsed['age']
    sex = parsed['sex']
    stage = parsed['stage']
    year = parsed['year']
    age_coef    = float(coefs_acs.loc[coefs_acs['name'] == 'age', 'value'].values[0])
    female_coef = float(coefs_acs.loc[coefs_acs['name'] == 'sex_female', 'value'].values[0])

    seer_coef_df = coefs_acs[coefs_acs['type'] == 'seer']
    seer_map = {int(n): float(v) for n, v in zip(seer_coef_df["name"], seer_coef_df["value"]) if str(n).isdigit()}

    lr = age_coef * age + female_coef * sex + seer_map.get(stage, 0.0)

    nyear = int(year) * 365
    row = basehazard_acs.loc[basehazard_acs['time'] == nyear, 'hazard']
    if row.empty:
        nearest_idx = (basehazard_acs['time'] - nyear).abs().idxmin()
        basehazard = float(basehazard_acs.loc[nearest_idx, 'hazard'])
    else:
        basehazard = float(row.values[0])
    prob = math.exp(-1.0 * basehazard * math.exp(lr))
    LOGGER.info(f"[NODE] run_calculation | model_id={model_id}, parsed={parsed}, prob={prob}")
    return {"survival_prob": prob}

    
def format_output(state: CalcState) -> OutputState:
    prob = state.get("survival_prob")
    extracted = state.get("extracted", {})

    if prob is None:
        return {"answer": "현재 입력된 변수 조합에 맞는 생존확률 계산 모델이 없습니다."}

    # ① 내부 모델 계산 결과
    base_msg = f"저희 내부 모델에 의한 5년 생존 확률은 약 {prob * 100:.1f}% 입니다."

    # ② RAG 검색
    sex_value = extracted.get("sex")
    sex_label = "여성" if sex_value == 1 else "남성" if sex_value == 0 else "전체"
    cancer_type = "위암"  # 현재는 예시, 나중에 extracted에서 가져올 수 있음
    year_range = "2018–2022"

    rag_result = get_cancer_survival_rate(cancer_type, sex_label, year_range)

    # ③ 최종 출력
    answer = (
        f"{base_msg}\n\n"
        f"📊 [국가암정보센터 참고]\n"
        f"{rag_result}\n\n"
        "※ 위 통계는 전체 위암 환자 집단의 평균값이며, "
        "실제 개인의 임상 상황에 따라 차이가 있을 수 있습니다."
    )

    LOGGER.info(f"[NODE] format_output | answer={answer}")
    return {"answer": answer}

# def parse_input(state: ModelSelectState) -> ModelSelectState:
#     query = state["raw_query"]
#     parsed = {"age": 60, "sex": "F", "stage": 2, "horizon": 5}
#     LOGGER.info(f"[NODE] parse_input | query={query} -> parsed={parsed}")
#     return {"parsed_input": parsed, "model_id": ""}


def run_calculation(state: ModelSelectState) -> CalcState:
    """내부 모델을 통한 생존확률 계산 함수 (model_7)."""
    model_id = state["model_id"]

    if model_id != "model_7":
        LOGGER.warning(f"[NODE] run_calculation | model_id={model_id} not supported.")
        return {"survival_prob": None}
    
    parsed = state["extracted"]
    age = parsed['age']
    sex = parsed['sex']
    stage = parsed['stage']
    year = parsed['year']
    age_coef    = float(coefs_acs.loc[coefs_acs['name'] == 'age', 'value'].values[0])
    female_coef = float(coefs_acs.loc[coefs_acs['name'] == 'sex_female', 'value'].values[0])

    seer_coef_df = coefs_acs[coefs_acs['type'] == 'seer']
    seer_map = {int(n): float(v) for n, v in zip(seer_coef_df["name"], seer_coef_df["value"]) if str(n).isdigit()}

    lr = age_coef * age + female_coef * sex + seer_map.get(stage, 0.0)

    nyear = int(year) * 365
    row = basehazard_acs.loc[basehazard_acs['time'] == nyear, 'hazard']
    if row.empty:
        nearest_idx = (basehazard_acs['time'] - nyear).abs().idxmin()
        basehazard = float(basehazard_acs.loc[nearest_idx, 'hazard'])
    else:
        basehazard = float(row.values[0])
    prob = math.exp(-1.0 * basehazard * math.exp(lr))
    LOGGER.info(f"[NODE] run_calculation | model_id={model_id}, parsed={parsed}, prob={prob}")
    return {"survival_prob": prob}

## 국가암통계 


def fetch_kcca_stat_text() -> str:
    """국가암정보센터 페이지에서 주요 문단을 가져오는 간단한 RAG 참조용 함수."""
    try:
        url = "https://www.cancer.go.kr/lay1/S1T648C650/contents.do"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        paras = soup.select("div.cont_txt p")
        texts = [p.get_text().strip() for p in paras if p.get_text().strip()]

        return " ".join(texts[:3]) if texts else "국가암정보센터 페이지에서 통계 정보를 불러오지 못했습니다."
    except Exception as e:
        return f"통계 정보 로드 중 오류 발생: {e}"


def format_output(state: CalcState) -> OutputState:
    prob = state.get("survival_prob")
    extracted = state.get("extracted", {})

    if prob is None:
        return {"answer": "현재 입력된 변수 조합에 맞는 생존확률 계산 모델이 없습니다."}

    # -------------------------
    # 내부 모델 결과 메시지
    # -------------------------
    base_msg = f"저희 내부 모델에 의한 5년 생존 확률은 약 {prob * 100:.1f}% 입니다."

    # -------------------------
    # RAG 검색 실행
    # -------------------------
    sex_value = extracted.get("sex")
    sex_label = "여성" if sex_value == 1 else "남성"
    seer_stage = extracted.get("stage")

    # 현재는 위암 고정 (원하면 cancer_type = extracted.get("cancer_type", "위암")로 일반화 가능)
    cancer_type = "위암"
    year_range = "2018–2022"

    try:
        stat_summary = get_cancer_survival_rate(cancer_type, sex_label, year_range, seer_stage)
    except Exception as e:
        stat_summary = f"국가암등록통계 검색 중 오류가 발생했습니다: {e}"

    # -------------------------
    # 결과 통합
    # -------------------------
    answer = (
        f"{base_msg}\n\n"
        f"[국가암등록통계 참고]\n"
        f"{stat_summary.strip()}\n\n"
        "※ 위 통계는 전체 위암 환자 집단의 평균값이며, 실제 개인의 임상 상황에 따라 차이가 있을 수 있습니다."
    )

    LOGGER.info(f"[NODE] format_output | answer={answer}")
    return {"answer": answer}

llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)

def general_chat(state: InputState) -> OutputState:
    query = state["raw_query"]
    answer = llm.invoke(query).content
    LOGGER.info(f"[NODE] general_chat | answer={answer}")
    return {"answer": answer}


# ======================
# 3. 그래프 구성
# ======================d
builder = StateGraph(OutputState, input_schema=InputState, output_schema=OutputState)

# 노드 추가
builder.add_node("classify_question", classify_question)
builder.add_node("variable_extraction", variable_extraction)
builder.add_node("select_model", select_model)
builder.add_node("run_calculation", run_calculation)
builder.add_node("format_output", format_output)
builder.add_node("general_chat", general_chat)  # 일반 대화 노드 추가

# 시작점 → 분류기로 이동
builder.add_edge(START, "classify_question")

# classify_question → 분기
builder.add_conditional_edges(
    "classify_question",
    lambda state: state["query_type"],  # query_type 값으로 분기
    {
        "survival": "variable_extraction",   # 생존분석 → parse_input으로
        "chat": "general_chat",      # 일반 대화 → general_chat으로
    },
)

# survival pipeline
builder.add_edge("variable_extraction","select_model")
builder.add_edge("select_model", "run_calculation")
builder.add_edge("run_calculation", "format_output")
builder.add_edge("format_output", END)

# chat pipeline
builder.add_edge("general_chat", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"raw_query": "60세 여성, seer stage 2, 5년 생존 확률??"})
    print(result)
    print(graph.get_graph().draw_mermaid())
