"""
녹색인증 FAQ 챗봇 - 검색 DB(chroma_db) 생성 스크립트

  - PDF 읽기   : PyPDFLoader (페이지 단위)
  - 빈 쪽 제외 : 표지·간지처럼 글자가 거의 없는 쪽은 조각으로 만들지 않음
  - 제목 붙이기: 조각마다 [분류]를, 질문 제목 없이 이어지는 쪽에는 [앞 질문]을 앞에 붙임
                 (예: 혜택 표가 이어지는 쪽 → "Q3 녹색인증에는 어떤 혜택이 있나요?"로 검색되도록)
  - 조각 나누기: RecursiveCharacterTextSplitter (최대 1000자, 겹침 100자)
  - 임베딩 모델: gemini-embedding-001 (app.py와 반드시 동일해야 함)

사용법
  1) 새 매뉴얼을 static 폴더에 manual.pdf 이름으로 둡니다(기존 파일 교체).
  2) 터미널에서 API 키를 지정합니다.
       Windows : set GOOGLE_API_KEY=발급받은키
       Mac     : export GOOGLE_API_KEY=발급받은키
  3) pip install -r requirements.txt
  4) python build_db.py
     (DB를 만들지 않고 쪽별 처리 결과만 미리 보려면: python build_db.py --preview)
  5) 새로 생긴 chroma_db 폴더를 GitHub에 올립니다(기존 폴더는 통째로 교체).
  6) app.py 상단의 MANUAL_NAME을 새 매뉴얼 이름으로 바꿉니다.
"""

import os
import re
import shutil
import sys
import time

PDF_PATH = "static/manual.pdf"
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_PAGE_CHARS = 60      # 이보다 글자가 적은 쪽(표지·간지)은 조각으로 만들지 않음 (app.py의 MIN_CHUNK_CHARS와 맞춤)
BATCH_SIZE = 20          # 무료 한도 대비: 한 번에 20개씩 나눠서 임베딩
WAIT_SECONDS = 5         # 배치 사이 대기 시간(초)

# 매뉴얼 구조 인식 규칙 (2025 매뉴얼 기준 — 새 매뉴얼은 --preview로 결과를 먼저 확인하세요)
SECTION_RE = re.compile(r"^(\d-[가-힣]\.\s*\S.*|\d\.\s+\S.*)$")   # 예: '1-나. 녹색인증의 혜택', '3. 녹색인증제도 케이스별 법률자문 모음'
SUBSECTION_RE = re.compile(r"^[①-⑳]\s*\S.*$")                     # 예: '① 지식재산권'
QUESTION_RE = re.compile(r"^Q\s?\d+\s+\S.*$")                      # 예: 'Q3 녹색인증에는 어떤 혜택이 있나요?'


def build_prefixes(texts):
    """쪽별 본문 목록을 받아, 각 쪽 조각 앞에 붙일 머리말을 돌려줌.
    None이면 그 쪽은 조각으로 만들지 않음(빈 쪽). 분류·질문 제목은 간지와 질문 줄에서 자동 인식."""
    section, subsection, last_q = "", "", ""
    prefixes = []
    for text in texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        body = text.strip()

        if len(body) < MIN_PAGE_CHARS:
            # 간지: 분류 정보만 갱신하고 조각은 만들지 않음
            # (한 간지에 대분류·소분류가 함께 있으면 줄 순서와 무관하게 대분류 → 소분류 순으로 반영)
            for ln in lines:
                if SECTION_RE.match(ln):
                    section, subsection, last_q = ln, "", ""
            for ln in lines:
                if SUBSECTION_RE.match(ln):
                    subsection, last_q = ln, ""
            prefixes.append(None)
            continue

        questions = [ln for ln in lines if QUESTION_RE.match(ln)]
        head = []
        if section:
            head.append(f"[분류: {section}{' > ' + subsection if subsection else ''}]")
        if not questions and last_q:
            head.append(f"[앞 질문에서 이어지는 내용: {last_q}]")
        prefixes.append("\n".join(head))

        if questions:
            last_q = questions[-1]
    return prefixes


def load_pages():
    from langchain_community.document_loaders import PyPDFLoader
    pages = PyPDFLoader(PDF_PATH).load()
    print(f"PDF {len(pages)}쪽을 읽었습니다.")
    return pages


def preview():
    """DB를 만들지 않고 쪽별 처리 결과만 출력 (API 키 불필요)."""
    pages = load_pages()
    for p, prefix in zip(pages, build_prefixes([p.page_content for p in pages])):
        label = p.metadata.get("page_label")
        if prefix is None:
            print(f"{label:>4}쪽  (제외: 빈 쪽/간지)")
        else:
            print(f"{label:>4}쪽  {prefix.replace(chr(10), ' ') or '(머리말 없음)'}")


def main():
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if "GOOGLE_API_KEY" not in os.environ:
        raise SystemExit("GOOGLE_API_KEY 환경변수를 먼저 설정해 주세요.")
    if not os.path.exists(PDF_PATH):
        raise SystemExit(f"{PDF_PATH} 파일을 찾을 수 없습니다.")

    # 1. 기존 DB 삭제 (이전 매뉴얼 내용과 섞이지 않도록)
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("기존 chroma_db 폴더를 삭제했습니다.")

    # 2. PDF 읽기 + 빈 쪽 제외·머리말 계산
    pages = load_pages()
    prefixes = build_prefixes([p.page_content for p in pages])
    kept = [(p, pre) for p, pre in zip(pages, prefixes) if pre is not None]
    print(f"빈 쪽·간지 {len(pages) - len(kept)}쪽을 제외하고 {len(kept)}쪽을 사용합니다.")
    if len(kept) < len(pages) * 0.5:
        print("주의: 글자가 거의 없는 쪽이 절반 이상입니다. 스캔(이미지) PDF일 수 있으니 확인하세요.")

    # 3. 조각 나누기 → 조각마다 머리말 붙이기 (긴 쪽이 여러 조각으로 나뉘어도 모두 붙음)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for page, prefix in kept:
        for chunk in splitter.split_documents([page]):
            if prefix:
                chunk.page_content = f"{prefix}\n{chunk.page_content}"
            chunks.append(chunk)
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


if __name__ == "__main__":
    if "--preview" in sys.argv:
        preview()
    else:
        main()
