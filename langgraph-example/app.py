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
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    # 그냥 LangGraph에 태움 (내부 classify_question이 분기 처리)
    result = graph.invoke({"raw_query": query, "query_type": None, "extracted": {}})
    ai_answer = result["answer"]

    add_to_memory(session_id, query, ai_answer)
    return JSONResponse({"answer": ai_answer})

@app.post("/reset_memory")
async def reset_memory(session_id: str = Form("default")):
    if session_id in SESSION_MEMORY:
        del SESSION_MEMORY[session_id]
        LOGGER.info(f"[MEMORY RESET] session={session_id}")
    return {"ok": True, "session_id": session_id, "message": "memory cleared"}

pdf_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse({"error": "PDF 파일만 업로드 가능합니다."}, status_code=400)

    # 1️⃣ PDF → 텍스트
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    if not text.strip():
        return JSONResponse({"error": "PDF에서 텍스트를 추출하지 못했습니다."}, status_code=400)

    # 2️⃣ 긴 문서 나누기
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    # 3️⃣ 부분 요약 프롬프트 (map 단계)
    map_prompt = ChatPromptTemplate.from_template(
        "다음 텍스트를 5문장 이내로 핵심만 요약해줘:\n\n{text}"
    )

    # 4️⃣ 전체 통합 요약 프롬프트 (reduce 단계)
    reduce_prompt = ChatPromptTemplate.from_template(
        "다음은 여러 부분 요약이야. 이를 기반으로 전체 내용을 한 문단으로 요약해줘:\n\n{summaries}"
    )

    # 5️⃣ 각 chunk별 요약 (map)
    partial_summaries = []
    for chunk in chunks:
        map_input = map_prompt.format_messages(text=chunk.page_content)
        summary_part = pdf_llm.invoke(map_input).content.strip()
        partial_summaries.append(summary_part)

    # 6️⃣ 전체 통합 (reduce)
    summaries_joined = "\n".join(partial_summaries)
    reduce_input = reduce_prompt.format_messages(summaries=summaries_joined)
    final_summary = pdf_llm.invoke(reduce_input).content.strip()

    return JSONResponse({"summary": final_summary})

