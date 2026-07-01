# College Smart Voice Assistant

Pure voice-based campus assistant for college information (Tamil + English + Tanglish).

## Features
- Voice input + browser speech output
- Noise reduction with Whisper transcription
- Wake-word activation: `Jarvis`
- Local SQLite database for departments and staff
- CSV bulk import for department data

## Run locally
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Recommended platform: Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud: https://share.streamlit.io
3. Connect your GitHub repo.
4. Select `app.py` as the main file.
5. Add `requirements.txt` to install dependencies.

> Note: Browser speech output uses the Web Speech API and works in modern browsers only.
