import os
import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from main import graph
from langchain_openai import ChatOpenAI

# =============================
# 1) Logger & API Key
# =============================
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
# 2) 세션별 memory
# =============================
SESSION_MEMORY = {}

def get_session_memory(session_id: str):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
    return SESSION_MEMORY[session_id]

def add_to_memory(session_id: str, user_msg: str, ai_msg: str):
    history = get_session_memory(session_id)
    history.append(HumanMessage(content=user_msg))
    history.append(AIMessage(content=ai_msg))
    LOGGER.info(f"[MEMORY] session={session_id}, user={user_msg}, ai={ai_msg}")

# =============================
# 3) FastAPI 앱
# =============================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

classifier = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def is_survival_question(text: str) -> bool:
    prompt = f"""
    다음 사용자의 질문이 '생존분석(survival analysis)' 혹은
    '생존 확률 계산'과 직접 관련이 있으면 YES, 아니면 NO로 답해.

    질문: "{text}"
    답변:
    """
    ans = classifier.invoke(prompt).content.strip().upper()
    return ans.startswith("Y")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(query: str = Form(...), session_id: str = Form("default")):
    LOGGER.info(f"[CHAT] session={session_id}, query={query}")
    history = get_session_memory(session_id)

    if is_survival_question(query):
        result = graph.invoke({"raw_query": query})
        ai_answer = result["answer"]
    else:
        general_llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
        messages = history + [HumanMessage(content=query)]
        ai_answer = general_llm.invoke(messages).content

    add_to_memory(session_id, query, ai_answer)
    return JSONResponse({"answer": ai_answer})

@app.post("/reset_memory")
async def reset_memory(session_id: str = Form("default")):
    if session_id in SESSION_MEMORY:
        del SESSION_MEMORY[session_id]
        LOGGER.info(f"[MEMORY RESET] session={session_id}")
    return {"ok": True, "session_id": session_id, "message": "memory cleared"}
