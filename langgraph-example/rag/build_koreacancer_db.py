import os
import re
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ======================
# 1️⃣ API 키 로드
# ======================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY and os.path.exists("api_key.txt"):
    with open("api_key.txt", "r", encoding="utf-8") as f:
        OPENAI_API_KEY = f.read().strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 api_key.txt를 확인하세요.")

# ======================
# 2️⃣ PDF 텍스트 추출
# ======================
pdf_path = "koreacancer.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")

doc = fitz.open(pdf_path)
text = "\n".join(page.get_text("text") for page in doc)

# 텍스트 정리 (표 구조 안정화)
text = re.sub(r"\s+", " ", text)
text = re.sub(r"(\d)\s*%\s*", r"\1%", text)
text = re.sub(r"’", "'", text)

print(f"✅ PDF에서 텍스트 추출 완료 (길이: {len(text)}자)")

# ======================
# 3️⃣ 문서 Chunk 분할
# ======================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 표 중심 문맥 유지
    chunk_overlap=100,    # 표 줄간 연결
    separators=["\n", "|", " "]
)
docs = splitter.create_documents([text])
print(f"📄 총 {len(docs)}개 문단으로 분할 완료")

# ======================
# 4️⃣ 임베딩 및 FAISS 벡터DB 생성
# ======================
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = FAISS.from_documents(docs, embeddings)

save_dir = "koreacancer_db"
os.makedirs(save_dir, exist_ok=True)
db.save_local(save_dir)

print(f"✅ 국가암등록통계 벡터DB 생성 완료 → {save_dir}/")
