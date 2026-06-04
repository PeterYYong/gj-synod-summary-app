# 나눔 회의록 요약 챗봇

Streamlit과 OpenAI Responses API를 사용해 나눔 1 / 나눔 2 회의록을 요약하고, 같은 세션 안에서 수정 요청을 이어갈 수 있는 웹앱입니다.

## 변경 사항

- 파일 업로드 기능을 제거했습니다.
- 회의록 원문은 텍스트 입력창에 붙여넣은 내용만 처리합니다.
- 세션 초기화 시 `st.session_state.mode_selector`를 수정하지 않도록 고쳤습니다.
- Streamlit 위젯 key 충돌을 피하기 위해 selectbox에는 별도 key를 쓰지 않고, text_area는 `input_version`으로 새 key를 생성합니다.
- 사용자에게 모델 파라미터를 노출하지 않습니다.
- 나눔 1 / 나눔 2 선택과 현재 세션 초기화만 사이드바에 둡니다.

## 폴더 구조

```text
project/
  app.py
  requirements.txt
  README.md
  prompts/
    nanum1_prompt.txt
    nanum2_prompt.txt
  .streamlit/
    secrets.toml.example
```

## 로컬 실행

```powershell
cd C:\Users\user\Downloads\GJ_Synod_Project_Package\nanum_summary_app

Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt --upgrade
python -m streamlit run app.py
```

## secrets 설정

`.streamlit/secrets.toml` 파일을 만들고 아래처럼 입력합니다.

```toml
OPENAI_API_KEY = "sk-proj-실제키"
APP_PASSWORD_4DIGIT = "1234"
OPENAI_MODEL = "gpt-5.5"
```

비용 절감을 원하면 `OPENAI_MODEL`을 `gpt-5.4` 또는 접근 가능한 저렴한 모델로 변경해 테스트하십시오.

## 주의

- 실제 `secrets.toml`은 GitHub에 올리지 마십시오.
- 회의록 원문은 서버 파일로 저장하지 않습니다.
- 사용자별 원문, 요약 결과, 수정 대화는 `st.session_state`에만 저장됩니다.
