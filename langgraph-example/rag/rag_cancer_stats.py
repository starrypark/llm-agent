import fitz
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# .env 로드 (현재 실행경로 기준 또는 상위 폴더에서 탐색)
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY and os.path.exists("api_key.txt"):
    with open("api_key.txt", "r", encoding="utf-8") as f:
        OPENAI_API_KEY = f.read().strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 api_key.txt를 확인하세요.")


# DB 경로 설정
DB_PATH = os.path.join(os.path.dirname(__file__), "koreacancer_db")

# LLM 및 임베딩 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# RAG Prompt
prompt = ChatPromptTemplate.from_template("""
당신은 한국의 국가암등록통계를 잘 아는 전문가입니다.
다음의 통계 내용을 참고하여, 질문에 대해 **수치 중심으로 정확하게** 대답하세요.
병기에 상관없이, 성별에 따른 생존율만 나타내세요.

<context>
{context}
</context>

질문: {question}

응답 형식:
- 암종: OO암
- 구간: OOOO–OOOO
- 성별: 남성/여성/전체
- 5년 상대생존율: OO.X%
- 추가 설명: (간단히 요약)
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 벡터DB 로드 
db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# RAG 체인 정의
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 외부에서 쓸 함수
def get_cancer_survival_rate(
    cancer_type: str, 
    sex_label: str, 
    year_range: str, 
    seer_stage: int | None = None
) -> str:
    """
    국가암정보센터 RAG 기반 생존율 검색 (병기 포함)
    예시:
        get_cancer_survival_rate("위암", "남성", "2018–2022", 3)
        get_cancer_survival_rate("폐암", "여성", "2018–2022")
    """

    # -------------------------
    # 병기 텍스트 매핑
    # -------------------------
    stage_text = ""
    stage_desc = ""
    if seer_stage is not None:
        try:
            stage_num = int(seer_stage)
            if stage_num == 1:
                stage_text, stage_desc = " 1기", " (국한 병기)"
            elif stage_num == 2:
                stage_text, stage_desc = " 2기", " (국소 병기)"
            elif stage_num == 3:
                stage_text, stage_desc = " 3기", " (원격 병기)"
            elif stage_num == 4:
                stage_text, stage_desc = " 4기", " (말기 또는 원격 병기)"
        except ValueError:
            pass  # stage가 숫자가 아닐 경우 무시

    # -------------------------
    # 쿼리 생성
    # -------------------------
    # 병기 키워드도 함께 포함 → 검색 정확도 향상
    query = (
        f"{year_range}년 {sex_label}의 {cancer_type}{stage_text} 5년 상대생존율"
        f"{stage_desc} 병기별 생존율 또는 요약병기별 생존율 정보 포함"
    )

    # -------------------------
    # RAG 호출
    # -------------------------
    try:
        result = rag_chain.invoke(query)
        return result
    except Exception as e:
        return f"⚠️ 국가암등록통계 검색 중 오류 발생: {e}"

