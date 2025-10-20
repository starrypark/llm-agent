from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
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
# DB 로드
db = FAISS.load_local("kcca_stats_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# 문서들 보기
texts = [doc.page_content for doc in db.docstore._dict.values()]
print(f"총 {len(texts)}개의 문서가 저장됨.\n")

for i, t in enumerate(texts[:5], start=1):
    print(f"--- 문서 {i} ---\n{t}\n")
