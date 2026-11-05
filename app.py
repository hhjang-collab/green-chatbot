import os
import re
import base64
from datetime import datetime, timezone, timedelta
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# =====================================================================
# 📌 [매년 확인] 매뉴얼·심의일정 설정 — 연도 관련 정보는 여기서만 관리합니다
# ---------------------------------------------------------------------
# ▸ 심의일정이 바뀌면: schedule.txt 내용만 교체하면 됩니다.
#     (첫 줄의 'OOOO년도'에서 연도를 자동으로 읽어 화면·답변에 반영)
# ▸ 매뉴얼이 바뀌면: static/manual.pdf 교체 → build_db.py 실행 → 아래 MANUAL_NAME 수정
#     + 답변 지침 9번의 절차·기간(결과 통보 10일 이내, 인증서 2~3주)이 새 매뉴얼과 같은지 확인
# =====================================================================
MANUAL_NAME = "2025 녹색인증 FAQ 매뉴얼"   # 화면 안내와 답변 출처 표기에 쓰이는 매뉴얼 이름
SCHEDULE_FILE = "schedule.txt"


def load_schedule():
    """일정표 내용과 연도를 읽어옴. 연도는 첫 줄의 'OOOO년도'에서 자동 인식."""
    if not os.path.exists(SCHEDULE_FILE):
        return "", ""
    with open(SCHEDULE_FILE, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"(\d{4})\s*년도", text)
    return text, (m.group(1) if m else "")


SCHEDULE_TEXT, SCHEDULE_YEAR = load_schedule()
SCHEDULE_LABEL = f"{SCHEDULE_YEAR}년도 녹색인증심의위원회 일정" if SCHEDULE_YEAR else "녹색인증심의위원회 일정"
# =====================================================================

# --- 🖼️ [이동 및 수정] 로고 변환 함수 (호출보다 위에 정의) ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""
# --------------------------------------------------------

# --- 🔗 답변 표시 보정: 링크·굵은 글씨가 한글 조사와 붙어 깨지는 현상 방지 ---
_URL_RE = re.compile(r"(?<!\]\()(?<!\[)(?<!<)https?://[A-Za-z0-9\-._~:/?#@!$&'+,;=%]+")

def tidy_markdown(text):
    # 1) 맨 URL을 [주소](주소) 형태로 바꿔, 뒤에 붙은 ')', '**', '에서' 등이 링크에 딸려 들어가지 않게 함
    def _to_link(m):
        raw = m.group(0)
        url = raw.rstrip(".,;:!?'")
        return f"[{url}]({url})" + raw[len(url):]
    text = _URL_RE.sub(_to_link, text)
    # 2) '**굵게(괄호)**에서'처럼 닫는 ** 바로 뒤에 글자가 오면 굵게가 풀리지 않는 문제 → 보이지 않는 공백 삽입
    text = re.sub(r"(?<=[^\s*])\*\*(?=[가-힣A-Za-z0-9])", "​**", text)
    return text
# --------------------------------------------------------

# 1. API 키 (스트림릿 금고에서 가져오기)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 2. 웹페이지 기본 설정 (항상 최상단에 위치)
st.set_page_config(page_title="녹색인증 FAQ 챗봇", page_icon="logo.png", layout="centered")

# --- 🎨 통합 CSS ---
st.markdown(
    """
    <style>
    /* 챗봇 대화창 내부 텍스트 밝기를 낮춰 눈의 피로 감소 */
    .stChatMessage p, .stChatMessage li {
        opacity: 0.85 !important;
    }
    /* 하위 목록·목록 안 문단은 투명도가 겹쳐 더 어두워지므로 겹침 방지 */
    .stChatMessage li li, .stChatMessage li p {
        opacity: 1 !important;
    }
    /* 'Press Enter to apply' 안내 문구 강제로 숨기기 */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    
    /* [수정] 투명도 효과 제거 및 로고 우측(right)으로 이동 유지 */
    .company-logo {
        position: fixed;
        top: 70px;      
        right: 30px;    /* 우측 상단 고정 */
        width: 110px;   
        z-index: 1000;  
        cursor: pointer;
    }
    
    /* 💡 [수정 1 제거] 사이드바 버튼 내부 텍스트 가운데 정렬로 복구 */
    /* 관련 CSS 코드를 제거하여 기본 가운데 정렬을 사용합니다. */

    /* 💡 [수정 2 제거 및 변경] 단순 링크 스타일 제거 */
    /* 하단 단순 밑줄 링크를 박스 버튼으로 변경하면서 관련 CSS를 제거했습니다. */
    
    /* 모바일 화면에서는 로고 작게 조절 (마찬가지로 우측 유지) */
    @media (max-width: 640px) {
        .company-logo {
            width: 80px;
            top: 60px;
            right: 10px; 
        }
    }
    </style>
    """, unsafe_allow_html=True
)
# -------------------------------------------------------------------

# --- 🖼️ [추가] 제작사 로고 화면에 띄우기 (HTML) ---
comp_img_base64 = get_base64_of_bin_file("company_logo.png") 

if comp_img_base64:
    st.markdown(
        f"""
        <a href="http://www.iptob.co.kr/" target="_blank" title="(주)아이피투비 홈페이지로 이동">
            <img src="data:image/png;base64,{comp_img_base64}" class="company-logo" alt="(주)아이피투비 로고">
        </a>
        """,
        unsafe_allow_html=True
    )
# -------------------------------------------------------------

# --- 🔒 비밀번호 잠금 기능 ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.warning("🔒 보안을 위해 비밀번호를 입력해주세요.")
    with st.form("login_form"):
        pwd = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("확인")
        
        if submitted:
            expected_pwd = st.secrets.get("APP_PASSWORD", "1234") 
            if pwd == expected_pwd:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("비밀번호가 일치하지 않습니다.")
    st.stop()
# -----------------------------

# 챗봇 로고 로딩
img_base64 = get_base64_of_bin_file("logo.png")
img_html = f'<img src="data:image/png;base64,{img_base64}" width="60" style="display: block;">' if img_base64 else ""

# --- 📝 헤더 영역 ---
header_html = f"""
<div style="text-align: center; width: 100%;">
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px;">
        {img_html}
        <h1 style="margin: 0; padding: 0; opacity: 0.85;">녹색인증 FAQ 매뉴얼 챗봇</h1>
    </div>
    <p style="margin-top: 10px; font-size: 1.05em; opacity: 0.75;">
        녹색인증 제도와 관련된 질문을 입력하시면, 매뉴얼을 기반으로 답변해 드립니다.
    </p>
    <p style="margin-top: -5px; font-size: 0.9em; opacity: 0.6;">
        ※ {MANUAL_NAME} 기준 (심의위원회 일정은 {SCHEDULE_YEAR}년 공지 반영)
    </p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# 간격을 위한 빈 줄
st.markdown("<br>", unsafe_allow_html=True)
# ----------------------------------------------------

# 3. 1초 만에 뇌(DB) 불러오기
@st.cache_resource
def load_rag():
    if not os.path.exists("./chroma_db"):
        st.error("🚨 앗! 지식 창고(chroma_db) 폴더를 찾을 수 없습니다. 깃허브에 폴더가 잘 올라갔는지 확인해주세요!")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # Gemini 3.x 모델은 temperature 등 설정을 빼고 기본값 사용을 권장 (구글 공식 문서)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    # 📅 심의위원회 일정 (상단 설정 블록에서 schedule.txt를 읽어옴)
    # 중괄호는 프롬프트 변수로 오인되므로 이스케이프 처리
    schedule_text = SCHEDULE_TEXT.replace("{", "{{").replace("}", "}}")
    sched_label = SCHEDULE_LABEL

    system_prompt = (
        "당신은 '녹색인증제도' 관련 질문에 답변하는 전문 AI 챗봇입니다.\n\n"
        "### 💡 [답변 행동 지침]\n"
        "1. 일상적인 인사: 사용자가 '안녕', '고마워' 등 인사를 건네면, 매뉴얼을 찾지 말고 '안녕하세요! 녹색인증 매뉴얼 챗봇입니다. 어떤 점이 궁금하신가요?'라고 가볍게 대답하세요.\n"
        "2. 정보 제공 방식: 녹색인증 관련 질문에는 불필요한 서론이나 미사여구를 빼고, 즉시 본론(핵심 정보)만 간결하게 답변하세요.\n"
        "3. 엄격한 정보 통제: 반드시 아래 제공된 [문맥(Context)] 정보만을 사용하여 답변하세요. 문맥에 없는 내용은 절대 지어내거나 유추하지 말고, '제공된 매뉴얼에서는 해당 내용을 찾을 수 없습니다.'라고 명확히 안내하세요.\n"
        "4. ⭐️ URL 및 링크 표기: 웹사이트 주소를 제공할 때는 괄호 ')'나 한글 조사(예: '에서', '를')가 URL 링크의 일부로 인식되지 않도록 주의하세요. 반드시 순수한 영문 주소만 띄어쓰기로 분리하여 링크로 처리하세요.\n"
        "5. ⭐️ 출처 표기 (매우 중요): \n"
        "   - 글머리 기호나 문단마다 반복해서 출처를 달면 화면이 지저분해지므로 절대 금지합니다.\n"
        "   - 전체 답변이 모두 끝난 **맨 마지막에 딱 한 번만** 출처를 표기하세요.\n"
        "   - 단, 여러 페이지의 정보를 조합한 경우, 사용자가 헷갈리지 않도록 아래 예시처럼 어떤 내용이 몇 페이지에 있는지 괄호 안에 요약해서 적어주세요.\n"
        "   [출처 표기 형식 예시 — ○○p는 문맥에 나온 실제 쪽수로 채우세요]\n"
        f"   📗 출처: {MANUAL_NAME} (전체 인증절차: ○○p, 세부 신청 단계: ○○p, ○○p)\n\n"
        "6. 가독성: 사용자가 읽기 편하도록 핵심 키워드는 **굵은 글씨**로 강조하고, 여러 항목을 나열할 때는 글머리 기호(-, *)를 사용하세요.\n\n"
        "7. ⭐️ 심의위원회 일정 (매우 중요):\n"
        f"   - 심의위원회 개최 월·일자·회차·연간 횟수, 미개최 월에 대한 질문은 반드시 아래 [{sched_label}]만 기준으로 답하세요.\n"
        "   - 매뉴얼(문맥)에 나오는 심의위원회 개최 횟수·미개최 월·일자(예: '연 10회(2, 8월 미개최)')는 매뉴얼 발간 당시 기준이므로 인용하거나 언급하지 마세요. 개최 횟수와 미개최 월도 일정표의 안내를 따르세요.\n"
        "   - 매뉴얼에는 '정확한 심의 일자는 홈페이지 공지사항을 확인하라'는 안내가 있지만, 이 챗봇은 그 공지 내용을 이미 갖고 있습니다. 홈페이지 확인으로 안내를 대신하지 말고, 반드시 일정표의 실제 날짜를 직접 제시하세요.\n"
        "   - 일정을 물으면 남은 일정(오늘 이후)을 먼저 강조하고, 연간 전체 일정도 함께 보여주세요.\n"
        "   [일정 질문 답변 형식 예시 — ○ 부분은 일정표와 오늘 날짜({today})를 기준으로 실제 값으로 채우세요]\n"
        "   - **심의위원회 개최**: ○○○○년 연 ○회 (○월·○월 미개최)\n"
        "     - 다음 심의: **○월 ○일(○)** 제○○○회\n"
        "     - 남은 일정: ○월 ○일(○), ○월 ○일(○)\n"
        "     - (연간 전체 일정 나열)\n"
        "   - **결과발표**: (매뉴얼 내용으로 작성)\n"
        f"   📅 출처: 녹색인증 홈페이지 공지사항 「{sched_label}」\n"
        f"   📗 출처: {MANUAL_NAME} (결과발표: ○○p)\n"
        "   - '다음 심의', '이번 달 심의'처럼 시점이 필요한 질문은 오늘 날짜({today})를 기준으로 판단하세요.\n"
        "   - 오늘 날짜가 일정표의 마지막 일정을 지났다면, 다음 연도 일정은 아직 반영되지 않았다고 밝히고 녹색인증 홈페이지 공지사항에서 확인하도록 안내하세요. 이미 지난 일정을 '다음 심의'로 안내하지 마세요.\n"
        "   - 신청 마감일처럼 일정표에 없는 내용은 추측하지 말고, 녹색인증 홈페이지 공지사항을 확인하도록 안내하세요.\n"
        "   - 결과발표 시기, 결과 확인 방법 등 일정표에 없는 절차는 매뉴얼(문맥) 내용으로 답하세요.\n"
        "   - 일정 정보를 사용한 경우 출처에 아래처럼 함께 표기하세요.\n"
        f"     📅 출처: 녹색인증 홈페이지 공지사항 「{sched_label}」\n"
        "   - 심의 일정과 관련 없는 질문에는 일정 정보를 덧붙이지 마세요.\n\n"
        "8. 녹색인증 홈페이지 주소는 반드시 https://www.greencertif.or.kr 로 표기하세요. 다른 주소를 만들어 쓰지 마세요.\n\n"
        "9. ⭐️ 소요기간·인증 예상 시점 질문 ('지금 신청하면 언제 인증받나요?', '얼마나 걸리나요?' 등):\n"
        "   - 매뉴얼에는 신청부터 인증까지의 전체 소요기간이 나와 있지 않습니다. 첫 문장에서 이 점을 분명히 밝히세요.\n"
        "   - 특정 날짜나 '약 ○개월' 같은 예상 기간을 계산하거나 추정해서 제시하지 마세요.\n"
        "   - 신청 건이 몇 월 심의위원회에 상정될지도 평가 진행 상황에 따라 달라지므로 단정하지 마세요.\n"
        "   - 대신 매뉴얼에서 확인되는 사실만 순서대로 정리하세요: 신청(상시 가능) → 서류 검토·보완 → 현장·종합평가 → 심의위원회 → 결과 통보(심의 후 10일 이내) → 인증서 수령(적합 판정 후 2~3주).\n"
        "   - 참고로 오늘({today}) 이후 남은 심의위원회 날짜를 짧게 덧붙이되, '해당 회차 상정 여부는 평가 진행에 따라 다름'을 함께 적으세요.\n"
        "   - 마지막에 정확한 일정은 담당 평가기관에 문의하도록 안내하세요.\n\n"
        f"[{sched_label}]\n{schedule_text}\n\n"
        "문맥(Context): {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

with st.spinner("시스템을 준비하고 있습니다..."):
    rag_chain = load_rag()

# --- 🎯 추천 질문 (왼쪽 사이드바 고정) ---
suggested_prompt = None

with st.sidebar:
    # 텍스트 링크와 구분선(<hr>)을 하나로 합치고 margin을 직접 통제합니다.
    st.markdown(
        """
        <div style="margin-top: 5px;">
            <a href="https://ip2b-work-tools.streamlit.app/" target="_blank" style="text-decoration: none; color: inherit; font-size: 15px; font-weight: 600;">
                🏠 홈으로
            </a>
        </div>
        <hr style="margin-top: 10px; margin-bottom: 15px; border: 0; border-top: 1px solid currentColor; opacity: 0.2;">
        """, 
        unsafe_allow_html=True
    )

    # --- (기존 코드) 원래 있던 사이드바 내용들 시작 ---
    st.markdown("### 💡 자주 묻는 질문")
    
    if st.button("🤔 녹색인증의 종류는?", use_container_width=True):
        suggested_prompt = "녹색인증의 종류는 무엇이 있나요?"
    if st.button("🎁 녹색인증의 혜택은?", use_container_width=True):
        suggested_prompt = "녹색인증에는 어떤 혜택이 있나요?"
    if st.button("🔄 전체 인증 절차는?", use_container_width=True):
        suggested_prompt = "전체 인증절차는 어떻게 되나요?"
    if st.button("📑 신청 구비 서류는?", use_container_width=True):
        suggested_prompt = "기술인증 신청 시 구비서류는 무엇인가요?"
    if st.button("⏳ 유효기간과 연장은?", use_container_width=True):
        suggested_prompt = "녹색인증 유효기간은 어떻게 되나요?"
    if st.button("💰 인증 평가 수수료는?", use_container_width=True):
        suggested_prompt = "평가수수료는 얼마인가요?"
    if st.button("🤝 기술, 제품 동시 신청은?", use_container_width=True):
        suggested_prompt = "동시신청이 가능한가요? 동시 신청 시 장점과 단점은 무엇인가요?"
    if st.button("🗓️ 심의일정 및 결과발표는?", use_container_width=True):
        suggested_prompt = f"{SCHEDULE_YEAR + '년 ' if SCHEDULE_YEAR else ''}심의위원회 일정과 결과발표 시기는 언제인가요?"

    # 기본 구분선 대신, 여백을 꽉 조인 HTML 구분선으로 교체!
    st.markdown(
        '<hr style="margin-top: 15px; margin-bottom: 15px; border: 0; border-top: 1px solid currentColor; opacity: 0.2;">', 
        unsafe_allow_html=True
    )
    
    # 💡 [수정 2] 관련 정보 섹션을 링크 박스 버튼 스타일로 변경
    st.markdown("### 🔗 관련 링크")
    
    # [설정]📗 녹색인증 FAQ 매뉴얼 링크 (원하는 주소를 아래에 입력하세요)
    # static 폴더의 PDF를 앱이 직접 제공 → 크롬 등 브라우저 PDF 뷰어로 바로 열림
    # (.streamlit/config.toml 의 enableStaticServing = true 필요)
    # Streamlit Community Cloud는 '/~/+/' 경로로 들어가야 앱 화면을 거치지 않고 파일에 바로 접근됨
    faq_manual_url = "https://green-chatbot-56eztzpmyzchahoppzgsmd.streamlit.app/~/+/app/static/manual.pdf"
    st.link_button(label="📗 녹색인증 FAQ 매뉴얼", url=faq_manual_url, use_container_width=True)
    
    # [설정]📞 전담·평가기관 연락처 링크 (원하는 주소를 아래에 입력하세요)
    contacts_url = "https://www.greencertif.or.kr/ptl/cDefinitionC/dedicated.do"
    st.link_button(label="📞 전담·평가기관 연락처", url=contacts_url, use_container_width=True)

    # 🧪 [임시] 설치된 패키지 버전 확인용 — requirements.txt 버전 고정 후 이 블록 삭제
    with st.expander("🧪 설치된 버전 확인 (임시)"):
        from importlib.metadata import version as _ver, PackageNotFoundError as _PNF
        _pkgs = ["streamlit", "langchain", "langchain-core", "langchain-community", "langchain-classic",
                 "langchain-google-genai", "langchain-text-splitters", "google-genai",
                 "google-ai-generativelanguage", "chromadb", "pypdf", "protobuf",
                 "opentelemetry-api", "opentelemetry-exporter-otlp-proto-grpc"]
        _lines = []
        for _p in _pkgs:
            try:
                _lines.append(f"{_p}=={_ver(_p)}")
            except _PNF:
                _lines.append(f"# {_p} (설치 안 됨)")
        st.code("\n".join(_lines), language=None)

# --- 🚦 오류 구분: 잠시 뒤 다시 하면 되는 일시적 오류인지 판별 ---
# 한도 초과(429)·서버 혼잡(503)·시간 초과는 '잠시 후 재시도' 안내, 그 외(키·모델·DB 문제 등)는 관리자 확인 필요
_TEMP_ERROR_KEYWORDS = ("429", "resource_exhausted", "resourceexhausted", "quota", "rate limit",
                        "503", "unavailable", "overloaded", "deadline", "timeout", "timed out")

def is_temporary_error(e):
    text = f"{type(e).__name__} {e}".lower()
    return any(k in text for k in _TEMP_ERROR_KEYWORDS)
# -------------------------------------------------------------------

# 4. 채팅 루프 (기존 기록 보여주기)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 사용자 입력창 및 답변 로직
prompt = st.chat_input("녹색인증제도에 대해 궁금한 점을 물어보세요!")
final_prompt = suggested_prompt or prompt 

if final_prompt:
    # 1. 사용자 질문 화면에 출력 (기록 저장은 답변 성공 후에 함께 → 실패 시 같은 질문이 중복으로 쌓이지 않음)
    with st.chat_message("user"):
        st.markdown(final_prompt)

    # 2. AI 답변 영역 (🚨 에러 방어막 추가!)
    with st.chat_message("assistant"):
        with st.spinner("답변을 준비하고 있습니다..."):
            try:
                # AI에게 매뉴얼 검색 및 답변을 요청합니다.
                today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y년 %m월 %d일")
                response = rag_chain.invoke({"input": final_prompt, "today": today})
                full_response = tidy_markdown(response["answer"])
                
                # 오류 없이 무사히 답변을 가져왔을 때만 화면에 출력합니다.
                st.markdown(full_response)
                
                # 성공한 경우에만 질문과 답변을 함께 대화 기록에 저장합니다.
                st.session_state.messages.append({"role": "user", "content": final_prompt})
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                if is_temporary_error(e):
                    # 💥 API 무료 한도 초과·서버 혼잡 등 잠시 뒤 해결되는 오류
                    st.warning("앗! 💦 현재 질문이 너무 많아 챗봇이 잠시 숨을 고르고 있습니다. 약 1분 뒤에 다시 시도해 주세요!")
                else:
                    # 🔧 API 키·모델 이름·DB 문제 등 기다려도 해결되지 않는 오류 → 원인 확인용 상세 내용 제공
                    st.error("답변을 만드는 중 오류가 발생했습니다. 잠시 후에도 계속되면 관리자에게 아래 상세 내용을 전달해 주세요.")
                    with st.expander("오류 상세 (관리자 확인용)"):
                        st.code(f"{type(e).__name__}: {e}", language=None)
