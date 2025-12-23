from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import requests  # API 통신용
import json      # JSON 처리용

import gradio as gr

# 파이프라인/설정/검색 (기존 라이브러리 유지)
from agent_core.infer import online_infer
from agent_core.rag import rag_search, rebuild_index
# from agent_core.config import ROOT

APP_TITLE = "SF5 STP 설변 에이전트 (로컬 ONLINE)"

# -----------------------------
# 설정 및 API 키
# -----------------------------
# [주의] API 키가 코드에 포함되어 있습니다. 보안에 유의하세요.
api_key = "AIzaSyDxSha1cgKCSxNl92sXcwXJvD8txA_h7SA"

# -----------------------------
# 유틸리티 함수
# -----------------------------
def _path_of(upload_obj) -> str:
    """Gradio File 형식(문자열/딕셔너리/객체) 어떤 것으로 와도 경로를 안전하게 추출."""
    if upload_obj is None:
        return ""
    if isinstance(upload_obj, dict) and "path" in upload_obj:
        return str(upload_obj["path"])
    p = getattr(upload_obj, "name", None)
    if isinstance(p, (str, os.PathLike)):
        return str(p)
    if isinstance(upload_obj, (str, os.PathLike)):
        return str(upload_obj)
    raise ValueError("업로드 파일 경로를 인식하지 못했습니다.")

def _read_html(path: str) -> str:
    if not path or not Path(path).exists():
        return "오버레이 HTML이 없습니다."
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"<pre>오버레이를 열 수 없습니다.\n{e}</pre>"

def _fmt_val(x, unit=""):
    if x is None:
        return "N/A"
    try:
        if unit == "min":
            return f"{float(x):.1f}분"
        if unit == "krw":
            return f"{int(x):,}원"
        return str(x)
    except Exception:
        return str(x)

def _summary_lines(res: Dict[str, Any]) -> List[str]:
    return [
        f"[{res.get('uid')}] 설계변경 분석 결과",
        f"- 총 패치 수: {res.get('n_patch','N/A')}",
        f"- 총 시간(예측): {_fmt_val(res.get('time_min_pred_uid'),'min')}",
        f"- 총 비용(예측): {_fmt_val(res.get('cost_krw_pred_uid'),'krw')}",
        f"- 총 시간(보정): {_fmt_val(res.get('time_min_calib_uid'),'min')}",
        f"- 총 비용(보정): {_fmt_val(res.get('cost_krw_calib_uid'),'krw')}",
    ]

# -----------------------------
# 생성형 Gemini API 함수 (REST API 방식 - Python 3.8 호환)
# -----------------------------
def generate_gemini_reply(user_text: str, context_text: str, api_key_val: str) -> str:
    """requests 라이브러리를 사용하여 Google Gemini API 직접 호출"""
    
    # 모델 설정 (가장 빠른 모델)
    model_name = "gemini-2.5-flash"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key_val}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # [수정됨] 시스템 프롬프트(페르소나) 추가
    # AI에게 '너는 누구인가'를 알려줘서 기계적인 번역투를 방지합니다.
    system_instruction = """
    당신은 'SF5 STP 설계 변경 분석 에이전트'입니다.
    사용자는 설계 엔지니어이며, STP 파일의 변경점(Before vs After)에 대해 질문할 것입니다.
    
    [지시사항]
    1. 항상 한국어로 자연스럽고 친절하게 대답하세요. (번역투 금지)
    2. 전문적인 용어를 사용하되 이해하기 쉽게 설명하세요.
    3. 인사는 "안녕하세요! 설계 변경 분석을 도와드릴까요?" 형태로 자연스럽게 하세요.
    4. 제공된 Context(분석 결과)가 있다면 그 내용을 바탕으로 구체적으로 답변하세요.
    """
    
    # 프롬프트 조합
    full_prompt = f"{system_instruction}\n\n[Context Data]\n{context_text}\n\n[User Question]\n{user_text}\n\n[Response]"
    
    payload = {
        "contents": [{
            "parts": [{"text": full_prompt}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            try:
                return data['candidates'][0]['content']['parts'][0]['text'].strip()
            except (KeyError, IndexError):
                return "AI 응답을 해석할 수 없습니다."
        else:
            return f"API 호출 오류 (Code {response.status_code}):\n{response.text}"
            
    except Exception as e:
        return f"통신 오류 발생: {str(e)}"

# -----------------------------
# 이벤트 핸들러
# -----------------------------
def on_run_pipeline(before_file, after_file, reuse_0108, make_09r_overlay, chat_history, last_result):
    before_path = _path_of(before_file)
    after_path  = _path_of(after_file)
    if not before_path or not after_path:
        msg = "입력 오류: Before/After STP를 모두 업로드하십시오."
        chat_history = chat_history + [("시스템", msg)]
        return chat_history, "N/A", gr.update(value="오버레이 HTML이 없습니다."), last_result

    try:
        res = online_infer(
            before_file=before_path,
            after_file=after_path,
            uid_request="AUTO",
            start_uid=151,
            reuse_0108_if_exists=bool(reuse_0108),
            make_overlay=bool(make_09r_overlay),
        ).to_dict()

        last_result = {
            "uid": res.get("uid"),
            "uid_num": res.get("uid_num"),
            "overlay_html": res.get("overlay_html"),
            "time_min_pred_uid": res.get("time_min_pred_uid"),
            "cost_krw_pred_uid": res.get("cost_krw_pred_uid"),
            "time_min_calib_uid": res.get("time_min_calib_uid"),
            "cost_krw_calib_uid": res.get("cost_krw_calib_uid"),
            "n_patch": res.get("n_patch"),
        }

        summary = "\n".join(_summary_lines(last_result))
        chat_history = chat_history + [("에이전트", summary)]

        html_text = _read_html(last_result.get("overlay_html"))
        return chat_history, str(last_result["uid"]), gr.update(value=html_text), last_result

    except Exception as e:
        chat_history = chat_history + [("시스템", f"[실패] 파이프라인 오류: {e}")]
        return chat_history, "N/A", gr.update(value=f"<pre>{e}</pre>"), last_result

def on_send_message(user_text, chat_history, last_result, llm_model, api_key_val):
    user_text = (user_text or "").strip()
    if not user_text:
        return chat_history, ""

    chat_history = chat_history + [("유재성님", user_text)]

    # 컨텍스트 구성 (분석 결과가 있으면 포함)
    context_text = ""
    if last_result:
        context_text = f"현재 분석된 UID 정보: {last_result}"

    try:
        # 모델 선택과 상관없이 내부적으로 gemini-2.5-flash를 사용하도록 고정함
        bot_response = generate_gemini_reply(user_text, context_text, api_key_val)
        bot_msg = bot_response
    except Exception as e:
        bot_msg = f"대화 응답 중 오류가 발생했습니다: {e}"

    chat_history = chat_history + [("에이전트", bot_msg)]
    return chat_history, ""

def on_rebuild_rag(chat_history):
    try:
        rebuild_index()
        chat_history = chat_history + [("시스템", "RAG 인덱스를 갱신했습니다.")]
        return chat_history
    except Exception as e:
        chat_history = chat_history + [("시스템", f"RAG 갱신 실패: {e}")]
        return chat_history

# -----------------------------
# UI 레이아웃
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft(), title=APP_TITLE) as demo:
    gr.Markdown(f"## {APP_TITLE}")

    last_result = gr.State(None)
    chat_history = gr.State([])
    
    # API 키 State
    api_key_state = gr.State(value=api_key)

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            llm_dd = gr.Dropdown(
                label="생성형 AI 선택",
                choices=["Gemini 2.5 Flash"],
                value="Gemini 2.5 Flash",
            )
            before = gr.File(label="Before STP", file_types=[".stp", ".STEP"], type="filepath")
            after  = gr.File(label="After STP",  file_types=[".stp", ".STEP"], type="filepath")

            with gr.Row():
                reuse_0108 = gr.Checkbox(label="01~08 재활용(이미 산출물 존재 시)", value=True)
                make09 = gr.Checkbox(label="09R Overlay HTML 생성", value=True)

            with gr.Row():
                run_btn = gr.Button("분석 실행", variant="primary")
                rebuild_btn = gr.Button("RAG 재빌드", variant="secondary")

            cur_uid = gr.Textbox(label="현재 UID", interactive=False)

        with gr.Column(scale=1):
            chat = gr.Chatbot(label="에이전트와 대화", bubble_full_width=False, height=520)
            user_txt = gr.Textbox(label="질문 입력", placeholder="예: 이 UID의 리브 변경 난이도와 공정 영향은?", lines=2)
            send_btn = gr.Button("질문 보내기", variant="secondary")

        with gr.Column(scale=2):
            viewer = gr.HTML(label="UID_xxx Overlay (09R)")

    run_btn.click(
        fn=on_run_pipeline,
        inputs=[before, after, reuse_0108, make09, chat_history, last_result],
        outputs=[chat, cur_uid, viewer, last_result],
        queue=False,
    )

    send_btn.click(
        fn=on_send_message,
        inputs=[user_txt, chat, last_result, llm_dd, api_key_state], 
        outputs=[chat, user_txt],
        queue=False,
    )

    rebuild_btn.click(
        fn=on_rebuild_rag,
        inputs=[chat],
        outputs=[chat],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True)