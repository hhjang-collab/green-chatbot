"""
녹색인증 FAQ 챗봇 - 검색 DB(chroma_db) 생성 스크립트

기존 2025 DB를 분석해 같은 방식으로 복원한 스크립트입니다.
  - PDF 읽기   : PyPDFLoader (페이지 단위)
  - 조각 나누기: RecursiveCharacterTextSplitter (최대 1000자, 겹침 100자)
  - 임베딩 모델: gemini-embedding-001 (app.py와 반드시 동일해야 함)

사용법
  1) 새 매뉴얼을 이 파일과 같은 폴더에 manual.pdf 이름으로 둡니다.
  2) 터미널에서 API 키를 지정합니다.
       Windows : set GOOGLE_API_KEY=발급받은키
       Mac     : export GOOGLE_API_KEY=발급받은키
  3) pip install -r requirements.txt
  4) python build_db.py
  5) 새로 생긴 chroma_db 폴더를 GitHub에 올립니다(기존 폴더는 통째로 교체).
"""

import os
import shutil
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "manual.pdf"
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BATCH_SIZE = 20          # 무료 한도 대비: 한 번에 20개씩 나눠서 임베딩
WAIT_SECONDS = 5         # 배치 사이 대기 시간(초)

if "GOOGLE_API_KEY" not in os.environ:
    raise SystemExit("GOOGLE_API_KEY 환경변수를 먼저 설정해 주세요.")
if not os.path.exists(PDF_PATH):
    raise SystemExit(f"{PDF_PATH} 파일을 찾을 수 없습니다.")

# 1. 기존 DB 삭제 (2025 내용과 섞이지 않도록)
if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)
    print("기존 chroma_db 폴더를 삭제했습니다.")

# 2. PDF 읽기
pages = PyPDFLoader(PDF_PATH).load()
print(f"PDF {len(pages)}쪽을 읽었습니다.")

empty = [p.metadata.get("page_label") for p in pages if len(p.page_content.strip()) < 20]
if len(empty) > len(pages) * 0.3:
    print(f"주의: 글자가 거의 없는 쪽이 {len(empty)}개입니다. 스캔 PDF일 수 있으니 확인하세요.")

# 3. 조각 나누기
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(pages)
print(f"조각 {len(chunks)}개로 나눴습니다.")

# 4. 임베딩 후 저장 (배치 단위, 실패 시 재시도)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

for start in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[start:start + BATCH_SIZE]
    for attempt in range(5):
        try:
            db.add_documents(batch)
            break
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"  오류 발생({e.__class__.__name__}), {wait}초 후 재시도합니다.")
            time.sleep(wait)
    else:
        raise SystemExit("재시도 5회 실패. 잠시 뒤 다시 실행해 주세요.")
    print(f"  {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} 저장 완료")
    time.sleep(WAIT_SECONDS)

# 5. 간단 검증
print("\n[검증] '평가수수료는 얼마인가요?' 검색 결과:")
for d in db.similarity_search("평가수수료는 얼마인가요?", k=3):
    print(f"  - {d.metadata.get('page_label')}쪽: {d.page_content[:60].strip()}...")

print("\n완료! chroma_db 폴더를 GitHub에 올려주세요.")
