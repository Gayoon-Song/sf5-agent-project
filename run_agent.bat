@echo off
:: 1. 프로젝트 폴더로 이동 (경로가 C:\sf5\sfsdh3 라고 가정)
cd /d C:\sf5\sfsdh3

:: 2. Conda 환경 활성화 
call conda activate my_project_env

:: 3. 웹 에이전트 실행
echo ==========================================
echo [SF5 Agent] Starting Server...
echo ==========================================
python app_gradio.py

:: 4. 서버 종료 시 대기
pause