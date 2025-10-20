from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
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
import pandas as pd

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY and os.path.exists("api_key.txt"):
    with open("api_key.txt", "r", encoding="utf-8") as f:
        OPENAI_API_KEY = f.read().strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 api_key.txt를 확인하세요.")

def build_kcca_db():
    url = "https://www.cancer.go.kr/lay1/S1T648C650/contents.do"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 여러 구조에서 문단 추출 시도
    candidates = []
    for selector in ["div.cont_txt p", "div.cont_text p", "section p", "div.contents p"]:
        candidates.extend([p.get_text().strip() for p in soup.select(selector) if p.get_text().strip()])

    texts = list(set(candidates))  # 중복 제거
    if not texts:
        raise ValueError("❌ 페이지에서 문단을 찾지 못했습니다. HTML 구조를 확인하세요.")

    # Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents(texts)

    # 안전장치: 빈 doc 확인
    if not docs:
        raise ValueError("❌ Splitter 결과가 비었습니다. 텍스트 확인 필요.")

    # 임베딩 및 저장
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("kcca_stats_db")

    print(f"✅ 국가암정보센터 통계 DB 구축 완료 ({len(docs)} chunks 저장됨)")

if __name__ == "__main__":
    build_kcca_db()