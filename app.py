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
st.set_page_config(page_title="녹색인증 FAQ 매뉴얼 챗봇", page_icon="logo.png", layout="centered")

# --- 🎨 통합 CSS (라이트/다크모드 호환, 눈부심 방지, Enter 문구 숨김, 제작사 로고 CSS) ---
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
    
/* [수정 후] 투명도 효과 제거 및 링크 클릭 커서 추가 */
    .company-logo {
        position: fixed;
        top: 70px;      /* 짤리지 않게 내린 위치 유지 */
        left: 30px;     
        width: 110px;   
        z-index: 1000;  
        cursor: pointer; /* 마우스 올리면 클릭 가능한 손가락 모양으로 바뀜 */
    }
    
    /* 모바일 화면에서는 로고 작게 조절 */
    @media (max-width: 640px) {
        .company-logo {
            width: 80px;
            top: 60px;
            left: 10px;
        }
    }
    </style>
    """, unsafe_allow_html=True
)
# -------------------------------------------------------------------

# --- 🖼️ [추가] 제작사 로고 화면에 띄우기 (HTML) ---
# 저장소에 올리신 제작사 로고 파일명으로 바꾸세요 (예: company_logo.png)
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
            expected_pwd = st.secrets.get("CHAT_PASSWORD", "1234") 
            if pwd == expected_pwd:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("비밀번호가 일치하지 않습니다.")
    st.stop()
# -----------------------------

# 챗봇 로고 로딩 (위 함수 이동으로 에러 해결됨)
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
    # ⭐️ 안전장치 추가: 만약 chroma_db 폴더가 없으면 에러 대신 안내문 띄우기
    if not os.path.exists("./chroma_db"):
        st.error("🚨 앗! 지식 창고(chroma_db) 폴더를 찾을 수 없습니다. 깃허브에 폴더가 잘 올라갔는지 확인해주세요!")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # 사용자님 토큰 사정에 맞게 gemini-1.5-flash나 gemini-pro로 수정해서 쓰세요!
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    
    system_prompt = (
        "당신은 '녹색인증제도' 관련 질문에 답변하는 전문 AI 챗봇입니다.\n\n"
        "### 💡 [답변 행동 지침]\n"
        "1. 일상적인 인사: 사용자가 '안녕', '고마워' 등 인사를 건네면, 매뉴얼을 찾지 말고 '안녕하세요! 녹색인증 매뉴얼 챗봇입니다. 어떤 점이 궁금하신가요?'라고 가볍게 대답하세요.\n"
        "2. 정보 제공 방식: 녹색인증 관련 질문에는 불필요한 서론이나 미사여구를 빼고, 즉시 본론(핵심 정보)만 간결하게 답변하세요.\n"
        "3. 엄격한 정보 통제: 반드시 아래 제공된 [문맥(Context)] 정보만을 사용하여 답변하세요. 문맥에 없는 내용은 절대 지어내거나 유추하지 말고, '제공된 매뉴얼에서는 해당 내용을 찾을 수 없습니다.'라고 명확히 안내하세요.\n"
        "4. ⭐️ 출처 표기 (매우 중요): \n"
        "   - 전체 답변 맨 밑에 출처를 한꺼번에 뭉쳐서 적지 마세요. 각 문장 끝에 (11)처럼 반복해서 달지도 마세요.\n"
        "   - 같은 내용(섹션)이 끝날 때마다 하단에 개별적으로 출처를 표기하세요.\n"
        "   - 🚨 주의: 마지막 글머리 기호(-) 옆에 이어서 적지 말고, 반드시 **줄바꿈(엔터)을 한 번 한 뒤 독립된 줄에** 적어주세요.\n"
        "   [출처 표기 예시]\n"
        "   ### 연장 신청 방법 및 시기\n"
        "   - 유효기간 만료 3~6개월 전에 신청해야 합니다.\n"
        "   - 신청 경로는 홈페이지를 이용합니다.\n"
        "\n"
        "   📗 출처: 2025 녹색인증 FAQ 매뉴얼 42p\n\n"
        "5. 가독성: 사용자가 읽기 편하도록 핵심 키워드는 **굵은 글씨**로 강조하고, 여러 항목을 나열할 때는 글머리 기호(-, *)를 사용하세요.\n\n"
        "문맥(Context): {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

with st.spinner("시스템을 준비하고 있습니다..."):
    rag_chain = load_rag()

# --- 🎯 추천 질문 6개 ---
suggested_prompt = None
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🤔 녹색인증의 종류는?", use_container_width=True):
        suggested_prompt = "녹색인증의 종류는 무엇이 있나요?"
    if st.button("🎁 녹색인증의 혜택은?", use_container_width=True):
        suggested_prompt = "녹색인증에는 어떤 혜택이 있나요?"
        
with col2:
    if st.button("📑 신청 구비 서류는?", use_container_width=True):
        suggested_prompt = "기술인증 신청 시 구비서류는 무엇인가요?"
    if st.button("⏳ 유효기간과 연장은?", use_container_width=True):
        suggested_prompt = "녹색인증 유효기간은 어떻게 되나요?"
        
with col3:
    if st.button("💰 평가 수수료는?", use_container_width=True):
        suggested_prompt = "평가수수료는 얼마인가요?"
    if st.button("📋 종합평가 기준은?", use_container_width=True):
        suggested_prompt = "종합평가 시 평가기준은 무엇인가요?"

st.markdown("---") # 구분선

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
                # 시뻘건 에러 화면 대신, 노란색 경고창으로 예쁘게 안내합니다.
                st.warning("앗! 💦 현재 질문이 너무 많아 챗봇이 잠시 숨을 고르고 있습니다. 약 1분 뒤에 다시 시도해 주세요!")
                
                # 💡 꿀팁: 에러 안내문은 대화 기록(messages)에 저장하지 않습니다. 
                # 그래야 다음번 질문할 때 챗봇이 앞선 에러 때문에 헷갈려하지 않습니다.