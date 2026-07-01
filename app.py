import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import noisereduce as nr
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "college_data.db"
TEMP_AUDIO_PATH = BASE_DIR / "temp_audio.wav"
WAKE_WORD = "jarvis"

st.set_page_config(
    page_title="Campus Voice Assistant",
    page_icon="🎙️",
    layout="wide",
)
st.title("Campus Voice Assistant")
st.markdown("**Offline campus assistant for a fast, modern voice experience.**")

st.markdown(
    "<style>"
    "body {background: linear-gradient(135deg, #071019, #0b1f3b); color: #e2e8f0;}"
    "[data-testid='stAppViewContainer'] {background: transparent;}"
    "[data-testid='stHeader'] {background: transparent;}"
    "[data-testid='stSidebar'] {background-color: rgba(15, 23, 42, 0.96); border-radius: 16px;}"
    "section.main {padding-top: 0.5rem;}"
    ".css-1lcbmhc.e1fqkh3o1 {background: rgba(255,255,255,0.04); border-radius: 20px; padding: 1.1rem;}"
    "</style>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant_awake" not in st.session_state:
    st.session_state.assistant_awake = False

if "wake_word" not in st.session_state:
    st.session_state.wake_word = WAKE_WORD

if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# ====================== DATABASE ======================

def get_db_connection():
    return sqlite3.connect(DATABASE_PATH)


def init_db() -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY,
                dept_name TEXT,
                hod_name TEXT,
                total_seats INTEGER,
                filled_seats INTEGER,
                fees INTEGER,
                van_facility TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS staff (
                id INTEGER PRIMARY KEY,
                name TEXT,
                role TEXT,
                department TEXT,
                year TEXT,
                class_tutor_for TEXT
            )
            """
        )


def add_sample_data() -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM departments")
        if cursor.fetchone()[0] == 0:
            departments = [
                ("Computer Science", "Dr. Rajesh Kumar", 120, 85, 45000, "Yes - 2 vans"),
                ("Mechanical", "Dr. Priya Sharma", 60, 55, 40000, "Yes"),
                ("Civil", "Dr. Suresh", 60, 40, 38000, "No"),
            ]
            cursor.executemany(
                "INSERT INTO departments (dept_name, hod_name, total_seats, filled_seats, fees, van_facility) VALUES (?,?,?,?,?,?)",
                departments,
            )

            staff = [
                ("Mr. Arjun", "Tutor", "Computer Science", "1st Year", "CSE A"),
                ("Ms. Lakshmi", "Tutor", "Computer Science", "2nd Year", "CSE B"),
            ]
            cursor.executemany(
                "INSERT INTO staff (name, role, department, year, class_tutor_for) VALUES (?,?,?,?,?)",
                staff,
            )


init_db()
add_sample_data()

# ====================== VOICE PROCESSING ======================
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="float32")


def _write_temp_audio(audio_bytes: bytes) -> Path:
    TEMP_AUDIO_PATH.write_bytes(audio_bytes)
    return TEMP_AUDIO_PATH


def _read_audio_data(file_path: Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(file_path)
    return data, sample_rate


def _normalize_audio(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32, copy=False)
    if data.size == 0:
        return data
    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data = data / peak
    return data


def process_voice(audio_value, sample_rate: Optional[int] = None) -> str:
    if audio_value is None:
        return "No audio captured. Please try again."

    if isinstance(audio_value, np.ndarray):
        audio_data = audio_value.astype(np.float32, copy=False)
        rate = sample_rate or 16000
    else:
        if hasattr(audio_value, "getvalue"):
            audio_bytes = audio_value.getvalue()
        elif hasattr(audio_value, "getbuffer"):
            audio_bytes = audio_value.getbuffer()
        elif hasattr(audio_value, "read"):
            audio_bytes = audio_value.read()
        else:
            return "Unable to process audio input."

        if not isinstance(audio_bytes, (bytes, bytearray, memoryview)):
            return "Unable to process audio input."

        audio_path = _write_temp_audio(bytes(audio_bytes))
        try:
            audio_data, rate = _read_audio_data(audio_path)
        finally:
            if audio_path.exists():
                audio_path.unlink()

        audio_data = _normalize_audio(audio_data)

    if audio_data.size == 0:
        return "No valid speech detected. Please try again."

    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, stationary=False)
    reduced_noise = np.asarray(reduced_noise, dtype=np.float32)

    model = load_whisper_model()
    segments, _ = model.transcribe(
        reduced_noise,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"threshold": 0.35, "min_speech_duration_ms": 600},
        language=None,
        initial_prompt="college departments staff hod fees tutors seats admissions computer science mechanical civil"
    )

    transcription = " ".join(
        segment.text.strip() for segment in segments if getattr(segment, "text", "")
    ).strip()

    return transcription or "I did not understand that. Please speak clearly."

def render_speech(text: str) -> None:
    if not text.strip():
        return

    payload = json.dumps(text)
    html = f"""
    <script>
    const msg = new SpeechSynthesisUtterance({payload});
    msg.lang = 'en-US';
    msg.rate = 1.0;
    msg.pitch = 1.0;
    msg.volume = 1.0;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(html, height=1)


# ====================== QUERY ANSWER ======================
def answer_query(query):
    query_lower = query.lower().strip()
    conn = sqlite3.connect(DATABASE_PATH)
    
    if any(x in query_lower for x in ["thank", "thanks", "bye", "exit", "stop"]):
        st.session_state.assistant_awake = False
        return "Sure da, naan rajiyaaga iruppen. Appo innoru murai help venuma?"

    if any(x in query_lower for x in ["hod", "head", "தலைவர்"]):
        df = pd.read_sql("SELECT dept_name, hod_name FROM departments", conn)
        return "Here are the HOD details:\n" + df.to_string(index=False)
    
    elif any(x in query_lower for x in ["seat", "admission", "எத்தனை", "available"]):
        df = pd.read_sql("""SELECT dept_name, total_seats, filled_seats, 
                          (total_seats - filled_seats) as available 
                          FROM departments""", conn)
        return "Seat availability:\n" + df.to_string(index=False)
    
    elif any(x in query_lower for x in ["tutor", "staff", "டியூட்டர்", "teacher"]):
        df = pd.read_sql("SELECT * FROM staff", conn)
        return "Staff and tutors:\n" + df.to_string(index=False)
    
    elif any(x in query_lower for x in ["fees", "கட்டணம்", "fee"]):
        df = pd.read_sql("SELECT dept_name, fees FROM departments", conn)
        return "Fee details:\n" + df.to_string(index=False)

    elif any(x in query_lower for x in ["department", "departments", "department list", "அறை", "துறை", "which departments"]):
        df = pd.read_sql("SELECT dept_name FROM departments", conn)
        departments = ", ".join(df["dept_name"].tolist())
        return f"The available departments are: {departments}."
    
    else:
        return "Sorry da, enakku puriyala. You can ask about HOD, seats, staff, fees, or departments."

# ====================== MAIN UI ======================
st.subheader("🎤 Campus Voice Assistant")
st.write(
    "A professional campus voice assistant for offline college information. "
    "Say the wake word 'Jarvis', ask naturally, and receive a spoken response plus a summary."
)

wake_text = st.session_state.wake_word.capitalize()
status_col, input_col, summary_col = st.columns([1.2, 2.4, 1])

with status_col:
    st.markdown("### Assistant status")
    st.metric("State", "Active" if st.session_state.assistant_awake else "Sleeping")
    st.metric("Interactions", len(st.session_state.messages) // 2)
    st.markdown("**Wake word**")
    st.code(wake_text)
    if st.session_state.assistant_awake:
        st.success("Assistant is ready to respond.")
    else:
        st.info(f"Say '{wake_text}' to activate the assistant.")

with input_col:
    st.markdown("### Voice input")
    audio_value = st.audio_input("Record your question", key="voice_input")

with summary_col:
    st.markdown("### Last interaction")
    st.write("**Question**")
    st.write(st.session_state.last_user_query or "No question recorded yet.")
    st.write("**Answer**")
    st.write(st.session_state.last_response or "No response yet.")

if audio_value:
    with st.spinner("Processing voice input..."):
        user_query = process_voice(audio_value)
        st.session_state.last_user_query = user_query
        st.session_state.messages.append({"role": "user", "content": user_query})

        if st.session_state.wake_word in user_query.lower():
            st.session_state.assistant_awake = True
            response = "Hello, I am ready. How can I help you today?"
        elif not st.session_state.assistant_awake:
            response = (
                f"Please say '{wake_text}' first to activate the assistant. "
                f"For example: '{wake_text}, tell me the seat availability.'"
            )
        else:
            response = answer_query(user_query)

        st.session_state.last_response = response
        st.session_state.messages.append({"role": "assistant", "content": response})
        render_speech(response)

st.markdown("---")
with st.expander("Conversation History", expanded=True):
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='padding: 14px; margin-bottom: 10px; border-radius: 12px; background: #eef5ff;'>"
                f"<strong>You:</strong> {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='padding: 14px; margin-bottom: 10px; border-radius: 12px; background: #e6ffe6;'>"
                f"<strong>Assistant:</strong> {msg['content']}</div>",
                unsafe_allow_html=True,
            )

with st.sidebar:
    st.header("Assistant Controls")
    st.markdown(
        "Use the wake word **Jarvis** to activate the assistant. "
        "Ask about departments, faculty, seats, fees, or admissions."
    )
    st.markdown("---")

    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.assistant_awake = False
        st.session_state.last_user_query = ""
        st.session_state.last_response = ""
        st.experimental_rerun()

    if st.button("View database"):
        with get_db_connection() as conn:
            st.write("### Departments")
            st.dataframe(pd.read_sql("SELECT * FROM departments", conn))
            st.write("### Staff")
            st.dataframe(pd.read_sql("SELECT * FROM staff", conn))

    st.markdown("---")
    st.subheader("Import department data")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with get_db_connection() as conn:
            df.to_sql('departments', conn, if_exists='append', index=False)
        st.success("Department data imported successfully.")

    st.markdown("---")
    st.caption("Offline voice assistant · English + Tamil support")