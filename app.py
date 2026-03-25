import os
import base64
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 🖼️ [이동 및 수정] 로고 변환 함수 (호출보다 위에 정의) ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return "" 
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
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    
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
        "   [출처 표기 예시]\n"
        "   📗 출처: 2025 녹색인증 FAQ 매뉴얼 (전체 인증절차: 11p, 세부 신청 단계: 31p, 45p)\n\n"
        "6. 가독성: 사용자가 읽기 편하도록 핵심 키워드는 **굵은 글씨**로 강조하고, 여러 항목을 나열할 때는 글머리 기호(-, *)를 사용하세요.\n\n"
        "문맥(Context): {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

with st.spinner("시스템을 준비하고 있습니다..."):
    rag_chain = load_rag()

# --- 🎯 추천 질문 6개 (왼쪽 사이드바 고정) ---
suggested_prompt = None

with st.sidebar:
    st.markdown("### 💡 자주 묻는 질문")
    
    if st.button("🤔 녹색인증의 종류는?", use_container_width=True):
        suggested_prompt = "녹색인증의 종류는 무엇이 있나요?"
    if st.button("🎁 녹색인증의 혜택은?", use_container_width=True):
        suggested_prompt = "녹색인증에는 어떤 혜택이 있나요?"
    if st.button("📑 신청 구비 서류는?", use_container_width=True):
        suggested_prompt = "기술인증 신청 시 구비서류는 무엇인가요?"
    if st.button("⏳ 유효기간과 연장은?", use_container_width=True):
        suggested_prompt = "녹색인증 유효기간은 어떻게 되나요?"
    if st.button("💰 인증 평가 수수료는?", use_container_width=True):
        suggested_prompt = "평가수수료는 얼마인가요?"
    if st.button("📋 종합평가 기준은?", use_container_width=True):
        suggested_prompt = "종합평가 시 평가기준은 무엇인가요?"
# --- 추가된 추천 질문 4개 ---
    if st.button("🔄 전체 인증 절차는?", use_container_width=True):
        suggested_prompt = "전체 인증절차는 어떻게 되나요?"
    if st.button("🤝 기술/제품 동시 신청은?", use_container_width=True):
        suggested_prompt = "동시신청이 가능한가요? 동시 신청 시 장점과 단점은 무엇인가요?"
    if st.button("💸 수수료 환불 규정은?", use_container_width=True):
        suggested_prompt = "평가진행 중에 취소할 수 있나요? 수수료 반환은 가능한가요?"
    if st.button("🗓️ 결과 발표 및 심의일은?", use_container_width=True):
        suggested_prompt = "심의위원회 개최날짜와 결과발표는 언제인가요?"

    # (기존에 밖으로 빠져있던 구분선을 사이드바 안으로 넣었습니다)
    st.markdown("---") 
    
    # 💡 [수정 2] 관련 정보 섹션을 링크 박스 버튼 스타일로 변경
    st.markdown("### 🔗 관련 링크")
    
    # [설정]📗 2025 녹색인증 FAQ 매뉴얼 링크 (원하는 주소를 아래에 입력하세요)
    faq_manual_url = "https://cdn.jsdelivr.net/gh/hhjang-collab/green-chatbot-public@main/manual.pdf"
    st.link_button(label="📗 녹색인증 FAQ 매뉴얼", url=faq_manual_url, use_container_width=True)
    
    # [설정]📞 전담·평가기관 연락처 링크 (원하는 주소를 아래에 입력하세요)
    contacts_url = "https://www.greencertif.or.kr/ptl/cDefinitionC/dedicated.do"
    st.link_button(label="📞 전담·평가기관 연락처", url=contacts_url, use_container_width=True)

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
    # 1. 사용자 질문 화면에 출력
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    # 2. AI 답변 영역 (🚨 에러 방어막 추가!)
    with st.chat_message("assistant"):
        with st.spinner("답변을 준비하고 있습니다..."):
            try:
                # AI에게 매뉴얼 검색 및 답변을 요청합니다.
                response = rag_chain.invoke({"input": final_prompt})
                full_response = response["answer"]
                
                # 오류 없이 무사히 답변을 가져왔을 때만 화면에 출력합니다.
                st.markdown(full_response)
                
                # 성공한 답변만 대화 기록에 쏙 저장합니다.
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                # 💥 API 무료 한도 초과 등 예상치 못한 에러가 터졌을 때 실행되는 곳!
                st.warning("앗! 💦 현재 질문이 너무 많아 챗봇이 잠시 숨을 고르고 있습니다. 약 1분 뒤에 다시 시도해 주세요!")
