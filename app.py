import os
import uuid
import time
import random
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple

import streamlit as st
from openai import (
    OpenAI,
    RateLimitError,
    APIError,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
)

try:
    from docx import Document
except Exception:
    Document = None


# ============================================================
# 기본 설정
# ============================================================
APP_TITLE = "광주대교구 하느님 백성의 대화록 요약 AI 챗봇"
EVENT_TITLE = "제10차 하느님 백성의 대화"
EVENT_TOPIC = "소통이 교회를 살린다"

DEFAULT_MODEL = "gpt-5.5"

PROMPT_DIR = Path("prompts")
PROMPT_FILES = {
    "나눔 1": PROMPT_DIR / "nanum1_prompt.txt",
    "나눔 2": PROMPT_DIR / "nanum2_prompt.txt",
}
MODE_SLUG = {
    "나눔 1": "nanum1",
    "나눔 2": "nanum2",
}

# 행사 예상 동시 사용량을 고려해 내부 동시 API 요청 상한을 20으로 설정합니다.
# 사용자가 14명 정도라면 대기 없이 처리될 가능성이 높고,
# 실수성 중복 요청이 발생해도 무제한 폭주를 막을 수 있습니다.
OPENAI_MAX_CONCURRENT = 20
OPENAI_SEMAPHORE = threading.Semaphore(OPENAI_MAX_CONCURRENT)

# 한 세트 5,000자 기준. 과도한 전체 회의록 일괄 입력 방지용입니다.
MAX_INPUT_CHARS = 9000

# 사용자에게 노출하지 않는 고정 API 설정
TEXT_VERBOSITY = "medium"
REASONING_EFFORT = "medium"
REASONING_SUMMARY = "auto"
STORE_RESPONSES = False
DEFAULT_MAX_OUTPUT_TOKENS = 6000


COMMON_STYLE_RULES = """
공통 문체 보정 규칙:
1. 선택된 나눔 프롬프트의 구조와 요구사항을 우선 따르십시오.
2. 발표자가 실제로 읽어 줄 수 있는 자연스러운 한국어 문장으로 작성하십시오.
3. 발화자의 실제 단어와 중요한 문장 맥락은 유지하되, 반복되는 내용은 같은 의미끼리 묶어 정리하십시오.
4. 본문에는 발화자 이름, 번호, ID, 라벨을 직접 넣지 마십시오.
5. 원문에 없는 맥락, 결론, 감정, 담당자, 기한을 만들지 마십시오.
6. 같은 종결 표현을 연속해서 반복하지 마십시오.
7. “소개되었습니다”, “언급되었습니다”, “이야기가 나왔습니다”, “말이 있었습니다”는 필요할 때만 쓰고, 한 문단 안에서 반복하지 마십시오.
8. “반응이 있었습니다”, “반응으로 이어졌습니다”, “울림으로 이어졌습니다”, “공감이 형성되었습니다”는 사용하지 마십시오.
9. “그 마음에 함께 머물러 주셨습니다”, “그 어려움에 함께 마음을 모아 주셨습니다” 같은 인위적이고 감성적인 표현은 사용하지 마십시오.
10. 부정적 경험은 민원이나 비난처럼 쓰지 말고, 함께 살펴야 할 과제로 정리하십시오.
11. 기존 프롬프트에 참조표가 요구되어 있으면 참조표를 유지하십시오.
12. 참조표는 과도하게 길게 만들지 말고 핵심 근거 중심으로 작성하십시오. 특별한 지시가 없으면 전체 6~10행 정도로 제한하고, 원문이 매우 많아도 12행을 넘기지 마십시오.
13. 참조표가 필요한 경우에도 본문 가독성을 우선하고, 참조표에는 본문에 실제 반영된 핵심 표현만 적으십시오.

문체 재점검:
- 초안 작성 후, 문장이 너무 기계적으로 반복되는지 확인하십시오.
- “소개되었습니다/언급되었습니다/나왔습니다”가 반복되면 자연스러운 문장으로 다시 풀어 쓰십시오.
- 발표자가 소리 내어 읽었을 때 어색한 문장은 짧고 쉬운 문장으로 고치십시오.
- 원문에 없는 내용은 추가하지 마십시오.
""".strip()


# ============================================================
# 세션 상태 관리
# ============================================================
def init_session_state() -> None:
    """사용자별 세션 상태를 초기화합니다."""
    defaults = {
        "session_id": str(uuid.uuid4()),
        "authenticated": False,
        "selected_mode": "나눔 1",
        "raw_input_text": "",
        "combined_source_text": "",
        "current_summary": "",
        "chat_history": [],
        "prompt_version": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_version": 0,
        "is_generating": False,
        "is_revising": False,
        "pending_generation": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_current_session(keep_auth: bool = True) -> None:
    """
    현재 세션의 작업 데이터만 초기화합니다.

    Streamlit 위젯이 생성된 뒤 widget key를 직접 수정하면 오류가 날 수 있으므로,
    selectbox/text_area의 widget key는 직접 수정하지 않습니다.
    """
    authenticated = st.session_state.get("authenticated", False)
    selected_mode = st.session_state.get("selected_mode", "나눔 1")

    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state["selected_mode"] = selected_mode
    st.session_state["raw_input_text"] = ""
    st.session_state["combined_source_text"] = ""
    st.session_state["current_summary"] = ""
    st.session_state["chat_history"] = []
    st.session_state["prompt_version"] = ""
    st.session_state["created_at"] = datetime.now().isoformat(timespec="seconds")
    st.session_state["input_version"] = (
        int(st.session_state.get("input_version", 0)) + 1
    )
    st.session_state["is_generating"] = False
    st.session_state["is_revising"] = False
    st.session_state["pending_generation"] = False

    if keep_auth:
        st.session_state["authenticated"] = authenticated
    else:
        st.session_state["authenticated"] = False


# ============================================================
# secrets / 환경변수
# ============================================================
def get_secret_value(name: str, default: Optional[str] = None) -> Optional[str]:
    """Streamlit secrets를 우선 사용하고, 없으면 환경변수를 확인합니다."""
    try:
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
    except Exception:
        pass

    return os.environ.get(name, default)


def get_model_name() -> str:
    return get_secret_value("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def get_max_output_tokens() -> int:
    raw_value = get_secret_value(
        "OPENAI_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)
    )
    try:
        return int(raw_value)
    except Exception:
        return DEFAULT_MAX_OUTPUT_TOKENS


def get_openai_client() -> OpenAI:
    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. "
            "Streamlit Secrets 또는 서버 환경변수에 OPENAI_API_KEY를 설정하세요."
        )

    return OpenAI(api_key=api_key)


# ============================================================
# UI: 헤더 / 인증
# ============================================================
def render_header(auth_screen: bool = False) -> None:
    st.markdown(
        f"""
        <div style="
            padding: 1.1rem 1.2rem;
            border-radius: 1rem;
            background: linear-gradient(90deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #c7d2fe;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.95rem; color: #475569; font-weight: 600;">
                {EVENT_TITLE}
            </div>
            <div style="font-size: 1.8rem; color: #1e1b4b; font-weight: 800; margin-top: 0.2rem;">
                광주대교구 하느님 백성의 대화록 요약 AI 챗봇
            </div>
            <div style="font-size: 1rem; color: #334155; margin-top: 0.3rem;">
                주제: “{EVENT_TOPIC}”
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if auth_screen:
        return

    st.markdown(
        """
        <div style="
            padding: 0.9rem 1rem;
            border-radius: 0.75rem;
            background-color: #fff1f2;
            border: 2px solid #ef4444;
            color: #991b1b;
            font-weight: 700;
            margin-bottom: 1rem;
        ">
            중요: 나눔 1과 나눔 2는 프롬프트가 다릅니다. 입력 전 반드시 사이드바에서 요약 유형을 확인해 주세요.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            padding: 0.75rem 1rem;
            border-radius: 0.7rem;
            background-color: #fff7ed;
            border: 1px solid #fb923c;
            color: #9a3412;
            font-weight: 700;
            margin-bottom: 1rem;
        ">
            나눔 유형이 잘못 선택되면 요약문을 생성하지 않습니다.
            나눔 1은 경험과 마음에 남은 내용을, 나눔 2는 “내가/제가 ~라면, ~하겠습니다” 형식의 실천 다짐을 붙여넣어 주세요.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style="padding: 0.9rem; border-radius: 0.7rem; background-color: #f8fafc; border: 1px solid #e2e8f0;">
                <b>나눔 1</b><br>
                말하고 듣기 중심입니다. 참여자들이 나눈 경험, 어려움, 마음에 남은 표현을 정리합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style="padding: 0.9rem; border-radius: 0.7rem; background-color: #f8fafc; border: 1px solid #e2e8f0;">
                <b>나눔 2</b><br>
                “내가 본당/사제/수도자/봉사자라면, ~~ 하겠다” 형식의 실천 다짐을 정리합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="margin-top: 1rem; margin-bottom: 1rem; padding: 0.9rem; border-radius: 0.7rem; background-color: #ecfdf5; border: 1px solid #a7f3d0;">
            <b>활용법</b><br>
            1. 사이드바에서 <b>나눔 1 / 나눔 2</b>를 먼저 확인합니다.<br>
            2. 조별대화 원문을 입력창에 붙여넣습니다.<br>
            3. <b>요약 생성</b>을 한 번만 누르고 완료될 때까지 기다립니다.<br>
            4. 생성되는 요약문은 화면에 실시간으로 표시됩니다.<br>
            5. 결과가 나오면 채팅창에서 수정 요청을 이어갈 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_password() -> bool:
    expected = get_secret_value("APP_PASSWORD_4DIGIT")

    if not expected:
        st.error("APP_PASSWORD_4DIGIT가 설정되어 있지 않습니다. 관리자에게 문의하세요.")
        return False

    render_header(auth_screen=True)

    st.markdown(
        """
        <div style="padding: 1rem; border-radius: 0.7rem; background-color: #fff7ed; border: 1px solid #fdba74;">
            <b>접속 안내</b><br>
            관리자에게 받은 4자리 비밀번호를 입력해 주세요.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    with st.form("password_form"):
        password = st.text_input("4자리 비밀번호", type="password", max_chars=4)
        submitted = st.form_submit_button("접속")

    if submitted:
        if password == expected:
            st.session_state["authenticated"] = True
            st.success("인증되었습니다.")
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")

    return False


# ============================================================
# 프롬프트 로딩
# ============================================================
@st.cache_data(show_spinner=False)
def load_prompt_from_file(path_str: str) -> str:
    """
    프롬프트 파일은 고정 자료이므로 캐시합니다.
    사용자 원문, 요약 결과, 채팅 이력은 절대 캐시에 저장하지 않습니다.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"프롬프트 파일이 비어 있습니다: {path}")

    return text


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def load_prompt(mode: str) -> Tuple[str, str, str]:
    path = PROMPT_FILES[mode]
    text = load_prompt_from_file(str(path))
    version = f"{path.name}:{prompt_hash(text)}"
    return text, str(path), version


def build_instructions(mode: str, extra: str = "") -> Tuple[str, str]:
    """
    Prompt caching이 잘 작동하도록 고정 프롬프트를 instructions 앞쪽에 안정적으로 둡니다.
    사용자별 원문은 input에만 넣습니다.
    """
    prompt_text, prompt_name, version = load_prompt(mode)
    st.session_state["prompt_version"] = version

    parts = [
        prompt_text,
        "\n\n────────────────────────\n공통 보정 규칙\n────────────────────────\n",
        COMMON_STYLE_RULES,
    ]

    if extra.strip():
        parts.append(
            "\n\n────────────────────────\n추가 편집 지시\n────────────────────────\n"
        )
        parts.append(extra.strip())

    return "\n".join(parts), prompt_name


# ============================================================
# 나눔 유형 사전 검증
# ============================================================
def normalize_text_for_validation(text: str) -> str:
    """검증용으로 공백과 특수문자 영향을 줄입니다."""
    return " ".join((text or "").replace("\u3000", " ").split())


def count_contains(text: str, keywords: List[str]) -> int:
    """키워드가 포함된 횟수를 계산합니다."""
    return sum(text.count(keyword) for keyword in keywords)


def detect_nanum_type(source_text: str) -> Tuple[str, Dict[str, int]]:
    """
    붙여넣은 원문이 나눔 1에 가까운지, 나눔 2에 가까운지 로컬에서 판별합니다.
    API 호출 전 실행되므로 비용이 발생하지 않습니다.

    핵심:
    - 나눔 2는 “제가/내가 ~라면, ~하겠습니다” 역할 전환형 표현이 반복됩니다.
    - 나눔 1은 경험, 장면, 어려움, 마음에 남은 말, 공감/부담 중심입니다.
    """
    text = normalize_text_for_validation(source_text)

    nanum2_role_patterns = [
        "제가 수도자라면",
        "내가 수도자라면",
        "제가 성직자라면",
        "내가 성직자라면",
        "제가 사제라면",
        "내가 사제라면",
        "제가 평신도라면",
        "내가 평신도라면",
        "제가 봉사자라면",
        "내가 봉사자라면",
        "제가 본당이라면",
        "내가 본당이라면",
    ]

    nanum2_action_keywords = [
        "하겠습니다",
        "하겠다",
        "메신저",
        "의견을 더욱 물어보",
        "의견을 적극 말",
        "말할 책임",
        "책임 있게 전",
        "누가 빠졌는지",
        "익명으로 묶어 전달",
        "반복 가능한 통로",
    ]

    nanum1_experience_keywords = [
        "경험",
        "장면",
        "마음에 남",
        "서운",
        "어려움",
        "공유",
        "나누",
        "이야기",
        "소통",
        "참여",
        "상처",
        "실망",
        "기쁨",
        "위로",
        "존중",
        "불통",
        "좋은 시간",
        "주말",
        "참석 여부",
        "말하기 어려",
        "부담",
        "공감",
    ]

    nanum1_structure_keywords = [
        "1단계",
        "말하고 듣기",
        "2단계",
        "다시 듣고",
        "공감",
        "부담",
        "마음에 남",
    ]

    role_count = count_contains(text, nanum2_role_patterns)
    nanum2_action_count = count_contains(text, nanum2_action_keywords)
    nanum1_exp_count = count_contains(text, nanum1_experience_keywords)
    nanum1_structure_count = count_contains(text, nanum1_structure_keywords)

    nanum2_score = role_count * 5 + nanum2_action_count
    nanum1_score = nanum1_exp_count + nanum1_structure_count

    evidence = {
        "nanum1_score": nanum1_score,
        "nanum2_score": nanum2_score,
        "role_count": role_count,
        "nanum2_action_count": nanum2_action_count,
        "nanum1_exp_count": nanum1_exp_count,
        "nanum1_structure_count": nanum1_structure_count,
    }

    # 나눔 2는 역할 전환형 표현이 핵심이므로 강하게 판정합니다.
    if role_count >= 2 and nanum2_score >= 10:
        return "나눔 2", evidence

    # 나눔 1은 역할 전환형 표현이 거의 없어야 합니다.
    if role_count == 0 and nanum1_score >= 5:
        return "나눔 1", evidence

    # 역할 전환 표현이 1개라도 있고 나눔2 점수가 높으면 나눔2로 봅니다.
    if role_count >= 1 and nanum2_score > nanum1_score:
        return "나눔 2", evidence

    # 경험 중심 표현이 충분하고 역할 전환이 없으면 나눔1로 봅니다.
    if nanum1_score >= 7 and role_count == 0:
        return "나눔 1", evidence

    return "불명확", evidence


def validate_selected_nanum_mode(
    selected_mode: str, source_text: str
) -> Tuple[bool, str]:
    """선택한 나눔 유형과 붙여넣은 원문이 맞는지 확인합니다."""
    if not source_text or len(source_text.strip()) < 20:
        return False, "회의록 원문이 너무 짧거나 비어 있습니다."

    detected_type, evidence = detect_nanum_type(source_text)

    if detected_type == "불명확":
        return (
            False,
            (
                "붙여넣은 원문이 나눔 1인지 나눔 2인지 명확하지 않습니다.\n\n"
                "요약문을 생성하지 않았습니다.\n\n"
                "나눔 1은 경험·상황·마음에 남은 내용을 중심으로 입력하고, "
                "나눔 2는 “제가/내가 ~라면, ~하겠습니다” 형식의 역할별 실천 다짐을 입력해 주세요."
            ),
        )

    if detected_type != selected_mode:
        return (
            False,
            (
                f"선택이 잘못되었습니다.\n\n"
                f"현재 선택한 요약 유형은 [{selected_mode}]이지만, 붙여넣은 원문은 [{detected_type}]에 더 가깝습니다.\n\n"
                "요약문을 생성하지 않았습니다. 사이드바에서 올바른 나눔 유형을 선택한 뒤 다시 실행해 주세요.\n\n"
                f"판별 참고값: 나눔1 점수={evidence['nanum1_score']}, "
                f"나눔2 점수={evidence['nanum2_score']}, "
                f"역할 전환 표현 수={evidence['role_count']}"
            ),
        )

    return True, f"입력 검증 완료: [{selected_mode}] 원문으로 판단됩니다."


# ============================================================
# OpenAI 호출 안정화 + 실시간 스트리밍
# ============================================================
def response_to_text(response) -> str:
    """Responses API 응답에서 텍스트를 추출합니다."""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks: List[str] = []
    try:
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
    except Exception:
        pass

    return "\n".join(chunks).strip()


def extract_stream_delta(event) -> str:
    """Responses API stream event에서 텍스트 delta를 추출합니다."""
    event_type = getattr(event, "type", "")

    if event_type == "response.output_text.delta":
        return getattr(event, "delta", "") or ""

    # SDK 버전 차이에 대비
    delta = getattr(event, "delta", None)
    if isinstance(delta, str):
        return delta

    data = getattr(event, "data", None)
    if isinstance(data, dict):
        return str(data.get("delta") or data.get("text") or "")

    if data is not None:
        text = getattr(data, "delta", None) or getattr(data, "text", None)
        if text:
            return str(text)

    return ""


def build_response_kwargs(*, instructions: str, user_input: str) -> Dict:
    """Responses API 호출 인자를 구성합니다."""
    kwargs = {
        "model": get_model_name(),
        "instructions": instructions,
        "input": user_input,
        "text": {
            "format": {"type": "text"},
            "verbosity": TEXT_VERBOSITY,
        },
        "reasoning": {
            "effort": REASONING_EFFORT,
            "summary": REASONING_SUMMARY,
        },
        "store": STORE_RESPONSES,
    }

    max_output_tokens = get_max_output_tokens()
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens

    return kwargs


def create_response_non_streaming(*, instructions: str, user_input: str) -> str:
    """비스트리밍 fallback 호출입니다."""
    client = get_openai_client()
    kwargs = build_response_kwargs(instructions=instructions, user_input=user_input)

    try:
        response = client.responses.create(**kwargs)
    except Exception as exc:
        message = str(exc)
        if "max_output_tokens" in message or "Unsupported parameter" in message:
            kwargs.pop("max_output_tokens", None)
            response = client.responses.create(**kwargs)
        else:
            raise

    text = response_to_text(response)
    if not text:
        raise RuntimeError("모델 응답이 비어 있습니다. 다시 시도하세요.")

    return text


def create_response_streaming(
    *,
    instructions: str,
    user_input: str,
    output_placeholder,
) -> str:
    """
    스트리밍 응답을 실시간으로 화면에 표시합니다.
    실패 시 예외를 던지고 상위 함수에서 비스트리밍 fallback을 수행합니다.
    """
    client = get_openai_client()
    kwargs = build_response_kwargs(instructions=instructions, user_input=user_input)
    collected = ""

    # 최신 SDK의 responses.stream 우선 사용
    if hasattr(client.responses, "stream"):
        try:
            with client.responses.stream(**kwargs) as stream:
                for event in stream:
                    delta = extract_stream_delta(event)
                    if delta:
                        collected += delta
                        output_placeholder.markdown(collected + "▌")
                final_response = stream.get_final_response()
                final_text = response_to_text(final_response)
                if final_text:
                    collected = final_text
        except Exception as exc:
            message = str(exc)
            if "max_output_tokens" in message or "Unsupported parameter" in message:
                kwargs.pop("max_output_tokens", None)
                collected = ""
                with client.responses.stream(**kwargs) as stream:
                    for event in stream:
                        delta = extract_stream_delta(event)
                        if delta:
                            collected += delta
                            output_placeholder.markdown(collected + "▌")
                    final_response = stream.get_final_response()
                    final_text = response_to_text(final_response)
                    if final_text:
                        collected = final_text
            else:
                raise

        if collected.strip():
            output_placeholder.markdown(collected)
            return collected.strip()

    # 구버전 SDK 호환: create(stream=True)
    try:
        kwargs_with_stream = dict(kwargs)
        kwargs_with_stream["stream"] = True
        events = client.responses.create(**kwargs_with_stream)
        for event in events:
            delta = extract_stream_delta(event)
            if delta:
                collected += delta
                output_placeholder.markdown(collected + "▌")
    except Exception:
        raise

    if not collected.strip():
        raise RuntimeError("스트리밍 응답이 비어 있습니다.")

    output_placeholder.markdown(collected)
    return collected.strip()


def call_with_retry(call_fn: Callable[[], str], max_retries: int = 3) -> str:
    """
    API 오류와 rate limit에 대해 지수 backoff로 재시도합니다.
    인증 오류는 재시도하지 않습니다.
    """
    for attempt in range(max_retries):
        try:
            return call_fn()
        except AuthenticationError:
            raise
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = min(30, (2**attempt) + random.random())
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError, APIError):
            if attempt == max_retries - 1:
                raise
            wait = min(20, (2**attempt) + random.random())
            time.sleep(wait)

    raise RuntimeError("API 요청이 반복 실패했습니다. 잠시 후 다시 시도해 주세요.")


def safe_openai_call(call_fn: Callable[[], str]) -> str:
    """
    동시에 최대 20개 요청만 OpenAI API로 보냅니다.
    행사 규모에서는 대기 없이 처리될 가능성이 높고,
    실수성 폭주만 방지합니다.
    """
    with OPENAI_SEMAPHORE:
        return call_with_retry(call_fn)


def call_openai(
    *,
    instructions: str,
    user_input: str,
    output_placeholder,
    status_box=None,
) -> str:
    """OpenAI 호출 래퍼. 스트리밍 우선, 실패 시 비스트리밍 fallback."""

    def _call() -> str:
        if status_box is not None:
            status_box.write(
                f"동시 요청은 최대 {OPENAI_MAX_CONCURRENT}개까지 처리합니다. 요청이 많으면 잠시 대기할 수 있습니다."
            )
            status_box.write("요약문을 생성중입니다. 조금만 기다려 주세요.")
            status_box.write("생성되는 내용은 아래에 실시간으로 표시됩니다.")

        try:
            return create_response_streaming(
                instructions=instructions,
                user_input=user_input,
                output_placeholder=output_placeholder,
            )
        except Exception as exc:
            if status_box is not None:
                status_box.write(
                    f"실시간 표시가 지연되어 일반 응답으로 재시도합니다. ({type(exc).__name__})"
                )
            text = create_response_non_streaming(
                instructions=instructions,
                user_input=user_input,
            )
            output_placeholder.markdown(text)
            return text

    return safe_openai_call(_call)


# ============================================================
# 요약 / 수정 요청
# ============================================================
def generate_initial_summary(
    mode: str, source_text: str, output_placeholder, status_box=None
) -> str:
    instructions, _ = build_instructions(mode)

    user_input = f"""
다음 회의록 원문을 선택된 프롬프트와 공통 문체 보정 규칙에 따라 요약하십시오.

[회의록 원문]
{source_text}
""".strip()

    return call_openai(
        instructions=instructions,
        user_input=user_input,
        output_placeholder=output_placeholder,
        status_box=status_box,
    )


def needs_source_text_for_revision(user_request: str) -> bool:
    """
    문체 수정이면 원문 전체를 다시 보내지 않아 비용을 줄입니다.
    원문 검토가 필요한 요청만 source_text를 다시 보냅니다.
    """
    request = user_request.lower()
    keywords = [
        "원문",
        "누락",
        "빠진",
        "발화자",
        "참조표",
        "근거",
        "표현을 더 살려",
        "원문 표현",
        "다시 확인",
        "검토",
        "틀린",
        "잘못",
        "내용 추가",
        "반영 안",
    ]
    return any(keyword in request for keyword in keywords)


def revise_summary(
    mode: str,
    source_text: str,
    current_summary: str,
    user_request: str,
    output_placeholder,
    status_box=None,
) -> str:
    source_needed = needs_source_text_for_revision(user_request)

    extra = """
당신은 기존 요약본을 사용자의 수정 요청에 따라 다듬는 편집자입니다.
반드시 기존 요약본의 구조와 원문 근거를 유지하십시오.
사용자가 요청한 부분만 우선 수정하되, 문체 일관성이 필요하면 전체를 자연스럽게 정리하십시오.
원문에 없는 내용을 새로 만들지 마십시오.
기존 프롬프트가 참조표를 요구하면 참조표를 유지하십시오.
참조표는 과도하게 늘리지 말고 기존 행 수를 유지하거나 핵심 행만 남기십시오.
""".strip()

    instructions, _ = build_instructions(mode, extra=extra)

    if source_needed:
        user_input = f"""
[원문 회의록]
{source_text}

[현재 요약본]
{current_summary}

[이번 사용자 수정 요청]
{user_request}

위 내용을 바탕으로 수정된 최종 요약본을 작성하십시오.
""".strip()
    else:
        user_input = f"""
[현재 요약본]
{current_summary}

[이번 사용자 수정 요청]
{user_request}

원문 전체 재검토가 필요한 요청이 아닙니다.
현재 요약본의 구조와 의미를 유지하면서 사용자의 요청에 맞게 문체와 표현을 자연스럽게 수정하십시오.
원문에 없는 내용을 새로 만들지 마십시오.
참조표가 있으면 유지하되 행을 새로 늘리지 마십시오.
""".strip()

    if status_box is not None:
        if source_needed:
            status_box.write(
                "원문 재검토가 필요한 수정 요청으로 판단하여 원문과 함께 수정합니다."
            )
        else:
            status_box.write(
                "문체 수정 중심 요청으로 판단하여 원문 전체 재전송 없이 현재 요약본 중심으로 수정합니다."
            )

    return call_openai(
        instructions=instructions,
        user_input=user_input,
        output_placeholder=output_placeholder,
        status_box=status_box,
    )


# ============================================================
# 다운로드
# ============================================================
def make_download_file_name(mode: str, ext: str) -> str:
    slug = MODE_SLUG.get(mode, "nanum")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{slug}_summary_{ts}.{ext}"


def make_txt_download(text: str) -> bytes:
    return text.encode("utf-8")


def make_markdown_download(text: str) -> bytes:
    return text.encode("utf-8")


def make_docx_download(text: str) -> bytes:
    if Document is None:
        raise RuntimeError("python-docx가 설치되어 있지 않습니다.")

    doc = Document()
    doc.add_heading("Nanum Meeting Summary", level=1)

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue

        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("#### "):
            doc.add_heading(line[5:].strip(), level=4)
        elif line.startswith("##### "):
            doc.add_paragraph(line[6:].strip())
        elif line.startswith("- "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        else:
            doc.add_paragraph(line)

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()


# ============================================================
# UI
# ============================================================
def render_sidebar() -> str:
    st.sidebar.title("설정")

    if st.session_state.get("authenticated"):
        st.sidebar.success("인증됨")
    else:
        st.sidebar.warning("인증 필요")

    mode_options = ["나눔 1", "나눔 2"]
    current_mode = st.session_state.get("selected_mode", "나눔 1")
    default_index = (
        mode_options.index(current_mode) if current_mode in mode_options else 0
    )

    chosen_mode = st.sidebar.selectbox(
        "요약 유형",
        options=mode_options,
        index=default_index,
    )
    st.session_state["selected_mode"] = chosen_mode

    if st.sidebar.button("현재 세션 초기화", use_container_width=True):
        reset_current_session(keep_auth=True)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(f"모델: {get_model_name()}")
    st.sidebar.caption(f"동시 API 요청 제한: 최대 {OPENAI_MAX_CONCURRENT}개")
    st.sidebar.caption("입력 방식: 텍스트 붙여넣기 전용")

    try:
        prompt_text, prompt_name, _ = load_prompt(chosen_mode)
        with st.sidebar.expander("선택된 프롬프트 미리보기"):
            st.caption(prompt_name)
            st.text_area(
                "프롬프트 내용",
                value=prompt_text,
                height=260,
                disabled=True,
                key=f"prompt_preview_{chosen_mode}",
            )
    except Exception as exc:
        st.sidebar.error(str(exc))

    return chosen_mode


def render_input_area() -> str:
    st.subheader("1. 조별대화 원문 붙여넣기")
    st.caption(
        "파일 업로드는 사용하지 않습니다. 조별대화 원문을 그대로 붙여넣어 주세요."
    )

    input_key = f"raw_input_area_{st.session_state.get('input_version', 0)}"
    raw_text = st.text_area(
        "회의록 원문",
        height=360,
        placeholder="여기에 조별대화 원문을 붙여넣으십시오.",
        key=input_key,
    )

    st.session_state["raw_input_text"] = raw_text
    st.session_state["combined_source_text"] = raw_text.strip()

    if raw_text.strip():
        char_count = len(raw_text.strip())
        st.caption(f"입력 글자 수: {char_count:,}자 / 권장 최대 {MAX_INPUT_CHARS:,}자")

        if char_count > MAX_INPUT_CHARS:
            st.warning(
                f"입력문이 {MAX_INPUT_CHARS:,}자를 초과했습니다. "
                "행사 안정성을 위해 조별대화 1개 단위로 나누어 입력해 주세요."
            )

    return raw_text.strip()


def queue_summary_generation() -> None:
    """
    요약 생성 버튼 클릭 즉시 실행되는 callback입니다.

    Streamlit은 버튼 클릭 callback이 먼저 실행되고 그 뒤 스크립트를 다시 실행합니다.
    이 callback에서 상태를 먼저 바꿔 두면 다음 rerun에서 버튼이 즉시 비활성화된 상태로 렌더링됩니다.
    """
    if st.session_state.get("is_generating", False):
        return
    st.session_state["pending_generation"] = True
    st.session_state["is_generating"] = True


def render_summary_area(mode: str) -> None:
    st.subheader("2. 요약 생성")

    source_text = st.session_state.get("combined_source_text", "")
    too_long = len(source_text) > MAX_INPUT_CHARS
    is_generating = st.session_state.get("is_generating", False)
    pending_generation = st.session_state.get("pending_generation", False)

    if is_generating:
        st.markdown(
            """
            <div style="
                padding: 0.9rem 1rem;
                border-radius: 0.75rem;
                background-color: #eff6ff;
                border: 2px solid #2563eb;
                color: #1e3a8a;
                font-weight: 800;
                margin-bottom: 0.8rem;
            ">
                ⏳ 요약문을 생성중입니다. 조금만 기다려 주세요. 새로고침하거나 버튼을 다시 누르지 마세요.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 버튼은 callback으로 먼저 상태를 바꿉니다.
    # 이렇게 해야 다음 rerun에서 버튼이 즉시 비활성화되어 중복 클릭을 막을 수 있습니다.
    st.button(
        "요약 생성 중..." if is_generating else "요약 생성",
        type="primary",
        use_container_width=True,
        disabled=is_generating or too_long or not source_text.strip(),
        on_click=queue_summary_generation,
    )

    if too_long:
        st.warning(
            f"입력문이 {MAX_INPUT_CHARS:,}자를 초과했습니다. "
            "행사 안정성을 위해 조별대화 1개 단위로 나누어 입력해 주세요."
        )

    # callback으로 접수된 요청만 여기에서 처리합니다.
    # 사용자가 버튼을 누른 run과 실제 API 처리 run을 분리해 버튼 비활성화가 화면에 확실히 보이게 합니다.
    if pending_generation:
        if not source_text or len(source_text) < 20:
            st.session_state["pending_generation"] = False
            st.session_state["is_generating"] = False
            st.warning("요약할 회의록 원문이 너무 짧거나 비어 있습니다.")
            return

        if too_long:
            st.session_state["pending_generation"] = False
            st.session_state["is_generating"] = False
            return

        is_valid_mode, validation_message = validate_selected_nanum_mode(
            mode, source_text
        )

        if not is_valid_mode:
            st.session_state["pending_generation"] = False
            st.session_state["is_generating"] = False
            st.error(validation_message)
            st.info("API 호출을 하지 않았으므로 비용이 발생하지 않았습니다.")
            return

        st.success(validation_message)

        # 같은 요청이 rerun으로 중복 처리되지 않도록 API 호출 직전에 pending flag를 내립니다.
        st.session_state["pending_generation"] = False
        live_output = st.empty()

        try:
            with st.status(
                "요약문을 생성중입니다. 조금만 기다려 주세요.", expanded=True
            ) as status:
                status.write("나눔 유형 검증이 완료되었습니다.")
                status.write(
                    "요약 생성 요청이 접수되었습니다. 버튼은 생성이 끝날 때까지 비활성화됩니다."
                )
                status.write(
                    f"요청이 몰릴 경우 최대 {OPENAI_MAX_CONCURRENT}개까지 동시에 처리합니다."
                )
                status.write("생성되는 요약문은 아래에 실시간으로 표시됩니다.")

                summary = generate_initial_summary(
                    mode=mode,
                    source_text=source_text,
                    output_placeholder=live_output,
                    status_box=status,
                )

                status.write("요약 생성이 완료되었습니다.")
                status.update(label="요약 생성 완료", state="complete")

            st.session_state["current_summary"] = summary
            st.session_state["chat_history"] = [
                {
                    "role": "assistant",
                    "content": summary,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            ]
            st.session_state["is_generating"] = False
            st.success("요약 생성 완료")
            st.rerun()

        except Exception as exc:
            st.session_state["is_generating"] = False
            st.session_state["pending_generation"] = False
            st.error(f"요약 생성 중 오류가 발생했습니다: {exc}")
            st.info(
                "잠시 후 다시 시도하세요. API 일시 오류 또는 네트워크 문제일 수 있습니다."
            )

    if st.session_state.get("current_summary"):
        st.subheader("3. 요약 결과")
        st.markdown(st.session_state["current_summary"])


def render_chat_area(mode: str) -> None:
    if not st.session_state.get("current_summary"):
        return

    st.subheader("4. 수정 요청")
    st.caption(
        "현재 요약본을 바탕으로 수정합니다. 단순 문체 수정은 원문 전체를 다시 보내지 않아 비용을 줄입니다."
    )

    for message in st.session_state.get("chat_history", []):
        role = message.get("role", "assistant")
        if role not in ("user", "assistant"):
            role = "assistant"
        with st.chat_message(role):
            st.markdown(message.get("content", ""))

    if st.session_state.get("is_revising"):
        st.info("현재 수정 요청을 처리 중입니다. 완료될 때까지 기다려 주세요.")
        return

    user_request = st.chat_input(
        "수정 요청을 입력하세요. 예: 조금 더 자연스럽게, 참조표는 유지하고 본문만 수정"
    )
    if user_request:
        st.session_state["is_revising"] = True
        st.session_state["chat_history"].append(
            {
                "role": "user",
                "content": user_request,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

        live_output = st.empty()

        try:
            with st.status(
                "수정 요청을 반영하는 중입니다. 조금만 기다려 주세요.", expanded=True
            ) as status:
                status.write("수정본이 생성되는 대로 아래에 실시간으로 표시됩니다.")

                revised = revise_summary(
                    mode=mode,
                    source_text=st.session_state.get("combined_source_text", ""),
                    current_summary=st.session_state.get("current_summary", ""),
                    user_request=user_request,
                    output_placeholder=live_output,
                    status_box=status,
                )

                status.write("수정본 생성이 완료되었습니다.")
                status.update(label="수정 완료", state="complete")

            st.session_state["current_summary"] = revised
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": revised,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            st.session_state["is_revising"] = False
            st.rerun()

        except Exception as exc:
            st.session_state["is_revising"] = False
            st.error(f"수정 중 오류가 발생했습니다: {exc}")
            st.info(
                "잠시 후 다시 시도하세요. API 일시 오류 또는 네트워크 문제일 수 있습니다."
            )


def render_download_area(mode: str) -> None:
    if not st.session_state.get("current_summary"):
        return

    st.subheader("5. 다운로드")
    summary = st.session_state["current_summary"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Markdown 다운로드",
            data=make_markdown_download(summary),
            file_name=make_download_file_name(mode, "md"),
            mime="text/markdown",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "TXT 다운로드",
            data=make_txt_download(summary),
            file_name=make_download_file_name(mode, "txt"),
            mime="text/plain",
            use_container_width=True,
        )

    with col3:
        try:
            docx_data = make_docx_download(summary)
            st.download_button(
                "DOCX 다운로드",
                data=docx_data,
                file_name=make_download_file_name(mode, "docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as exc:
            st.button("DOCX 다운로드 불가", disabled=True, use_container_width=True)
            st.caption(str(exc))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")
    init_session_state()

    if not st.session_state.get("authenticated"):
        check_password()
        return

    mode = render_sidebar()

    render_header(auth_screen=False)

    st.markdown(
        f"""
        <div style="
            padding: 0.85rem 1rem;
            border-radius: 0.7rem;
            background-color: #eff6ff;
            border: 1px solid #93c5fd;
            color: #1e3a8a;
            font-weight: 800;
            font-size: 1.05rem;
            margin-bottom: 1rem;
        ">
            현재 요약 유형: {mode}
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        load_prompt(mode)
    except Exception as exc:
        st.error(f"프롬프트 파일 오류: {exc}")
        st.stop()

    render_input_area()
    render_summary_area(mode)
    render_chat_area(mode)
    render_download_area(mode)


if __name__ == "__main__":
    main()
