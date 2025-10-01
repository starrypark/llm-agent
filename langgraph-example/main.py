import logging
import os
from dotenv import load_dotenv
from typing import TypedDict, Dict, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

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

# ======================
# 1. 상태 정의 (TypedDict)
# ======================
class InputState(TypedDict):
    raw_query: str   # 사용자가 입력한 자연어 질문
    query_type: Optional[str]  # 질문 유형 (예: '생존분석', '일반질문' 등)

class ModelSelectState(TypedDict):
    parsed_input: Dict   # 나이, 성별, stage, 기간 등 파싱된 구조화 데이터
    model_id: str        # 선택된 모델 이름

class CalcState(TypedDict):
    survival_prob: float

class OutputState(TypedDict):
    answer: str      # 최종 사용자 응답 메시지


# ======================
# 2. 노드 함수 정의
# ======================

classifier = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

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

## Retreiver 먼저 정의

# MODEL_DOCS = [
#     {"name": "model_1", "desc": "Uses only sex variable"},
#     {"name": "model_2", "desc": "Uses only age variable"},
#     {"name": "model_3", "desc": "Uses only seer stage variable"},
#     {"name": "model_4", "desc": "Uses sex and age variables"},
#     {"name": "model_5", "desc": "Uses sex and seer stage variables"},
#     {"name": "model_6", "desc": "Uses age and seer stage variables"},
#     {"name": "model_7", "desc": "Uses sex, age, and seer stage variables"},
# ]
# texts = [f"{d['name']}: {d['desc']}" for d in MODEL_DOCS]

# embeddings = OpenAIEmbeddings()
# model_db = FAISS.from_texts(texts, embeddings)
# model_retriever = model_db.as_retriever()


def parse_input(state: InputState) -> ModelSelectState:
    query = state["raw_query"]
    parsed = {"age": 60, "sex": "F", "stage": 2, "horizon": 5}
    LOGGER.info(f"[NODE] parse_input | query={query} -> parsed={parsed}")
    return {"parsed_input": parsed, "model_id": ""}

def select_model(state: ModelSelectState) -> ModelSelectState:
    parsed = state["parsed_input"]
    model_id = "model_7" if parsed["stage"] == 2 else "default_model"
    LOGGER.info(f"[NODE] select_model | stage={parsed['stage']} -> model_id={model_id}")
    return {"parsed_input": parsed, "model_id": model_id}

def run_calculation(state: ModelSelectState) -> CalcState:
    model_id = state["model_id"]
    parsed = state["parsed_input"]
    prob = 0.6628 if model_id == "model_7" else 0.5
    LOGGER.info(f"[NODE] run_calculation | model_id={model_id}, parsed={parsed}, prob={prob}")
    return {"survival_prob": prob}

def format_output(state: CalcState) -> OutputState:
    prob = state["survival_prob"]
    answer = f"5년 생존 확률은 약 {prob*100:.1f}% 입니다."
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
# ======================
builder = StateGraph(OutputState, input_schema=InputState, output_schema=OutputState)

# 노드 추가
builder.add_node("classify_question", classify_question)
builder.add_node("parse_input", parse_input)
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
        "survival": "parse_input",   # 생존분석 → parse_input으로
        "chat": "general_chat",      # 일반 대화 → general_chat으로
    },
)

# survival pipeline
builder.add_edge("parse_input", "select_model")
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
