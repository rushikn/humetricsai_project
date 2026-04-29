import cv2
import numpy as np
import time
import sqlite3
import os
from dotenv import load_dotenv
from deepface import DeepFace
from flask import Flask, jsonify, Response, render_template
from flask_cors import CORS
import google.generativeai as genai

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python.core import base_options
from auth import auth
from flask import session
from pdf_utils import generate_pdf
from email_utils import send_email

from pynput import mouse, keyboard
import threading


# =========================
# SWIN TRANSFORMER MODULE
# =========================

import torch
import torch.nn as nn
from torchvision import transforms

class SwinEmotionRecognizer:
    def __init__(self, model_path="swinmodel_p.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None

        self.emotions = [
    'very_low',
    'low',
    'medium',
    'high'
]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_model(self):
        try:
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
        except Exception as e:
            print("Swin model load warning:", e)
            self.model = None

    def predict(self, frame):
        if self.model is None:
            return None

        try:
            img = self.transform(frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img)
                _, pred = torch.max(outputs, 1)

            return self.emotions[pred.item()]

        except Exception as e:
            print("Swin inference warning:", e)
            return None


swin_emotion_engine = SwinEmotionRecognizer()
# =========================
# 🧠 affect TRANSFORMER MODULE
# =========================

import torch
import torch.nn as nn
from torchvision import transforms

class affectEmotionRecognizer:
    def __init__(self, model_path="affectnet_e.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None

        self.emotions = [
            'anger', 'contempt', 'disgust', 'fear',
            'happy', 'neutral', 'sad', 'surprise'
        ]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_model(self):
        try:
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
        except Exception as e:
            print("affect model load warning:", e)
            self.model = None

    def predict(self, frame):
        if self.model is None:
            return None

        try:
            img = self.transform(frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img)
                _, pred = torch.max(outputs, 1)

            return self.emotions[pred.item()]

        except Exception as e:
            print("affect inference warning:", e)
            return None


affect_emotion_engine = affectEmotionRecognizer()

# =========================
# 🔐 LOAD ENV (SECURE KEY)
# =========================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
print("API KEY:", API_KEY)  
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
def get_main_face(faces, w, h):
    if not faces:
        return None

    center_x, center_y = w // 2, h // 2

    best_face = None
    min_dist = float("inf")

    for face in faces:
        nose = face[1]
        x = int(nose.x * w)
        y = int(nose.y * h)

        dist = (x - center_x)**2 + (y - center_y)**2

        if dist < min_dist:
            min_dist = dist
            best_face = face

    return best_face
def draw_face_box(frame, landmarks, w, h):
    xs = [int(l.x * w) for l in landmarks]
    ys = [int(l.y * h) for l in landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
def get_ai_recommendation(stress, emotion, posture, screen_time):
    return get_ai_bot_recommendation(stress, emotion, posture, screen_time)


def get_ai_recommendation_bot(stress, emotion, posture, screen_time):

    manual_advice = get_ai_bot_recommendation(
        stress, emotion, posture, screen_time
    )

    if stress <= 50:
        return manual_advice

    prompt = f"""
    User stress: {stress}
    Emotion: {emotion}
    Posture: {posture}
    Screen time: {screen_time} seconds

    Give a short simple wellness suggestion.
    """

    try:
        response = model.generate_content(prompt)

        print("Gemini RAW RESPONSE:", response)
        print("Gemini TEXT:", response.text)

        if response and response.text:
            return response.text

        return manual_advice

    except Exception as e:
        print("Gemini ERROR:", e)

        return manual_advice
    
    
import random

def get_ai_bot_recommendation(stress, emotion, posture, screen_time):

    def pick(options):
        return random.choice(options)

    # LOW STRESS
    if stress <= 50:
        return pick([
            "You are doing well. Keep going 👍",
            "Everything looks fine. Stay focused!",
            "No stress detected. Keep it up!"
        ])

    # MEDIUM STRESS
    elif stress <= 70:

        if emotion in ["sad", "angry"]:
            return pick([
                "You seem a bit stressed. Take a short break.",
                "Try relaxing your mind for a few minutes.",
                "Pause for a moment and breathe slowly."
            ])

        if posture.startswith("Bad"):
            return pick([
                "Your posture looks off. Sit straight.",
                "Adjust your posture to avoid strain.",
                "Try sitting upright for better comfort."
            ])

        if screen_time > 1800:
            return pick([
                "You’ve been on screen too long. Rest your eyes.",
                "Look away from the screen for a few minutes.",
                "Give your eyes a short break."
            ])

        return pick([
            "Take a deep breath and stay calm.",
            "Relax for a minute before continuing.",
            "Keep going, but don’t overwork yourself."
        ])

    # HIGH STRESS
    else:

        if emotion == "angry":
            return pick([
                "You look frustrated. Pause and breathe.",
                "Take a break to calm down.",
                "Step away and relax your mind."
            ])

        if emotion == "sad":
            return pick([
                "You seem low. Take a short relaxing break.",
                "Do something you enjoy for a few minutes.",
                "Refresh your mind with a small break."
            ])

        if posture.startswith("Bad"):
            return pick([
                "Fix your posture immediately.",
                "Sit straight to avoid discomfort.",
                "Correct your posture now."
            ])

        if screen_time > 3600:
            return pick([
                "You’ve been working too long. Take a proper break.",
                "Walk around and rest your eyes.",
                "Step away from the screen for some time."
            ])

        return pick([
            "High stress detected. Take a break.",
            "Pause your work and relax.",
            "Give yourself some rest."
        ])
# =========================
# DATABASE
# =========================
conn = sqlite3.connect("stress.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    stress INTEGER,
    emotion TEXT,
    blinks INTEGER,
    screen_time INTEGER,
    mouse_clicks INTEGER,
    key_strokes INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS logins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    login_time DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    file_path TEXT,
    sent_time DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.commit()
# =========================
# USER TRACKING DATABASE (NEW)
# =========================
user_conn = sqlite3.connect("user_tracking.db", check_same_thread=False)
user_cursor = user_conn.cursor()

user_cursor.execute("""
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    stress INTEGER,
    session_start DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

user_conn.commit()
# =========================
# DAILY SESSION DATABASE (NEW)
# =========================
daily_conn = sqlite3.connect("daily_tracking.db", check_same_thread=False)
daily_cursor = daily_conn.cursor()

daily_cursor.execute("""
CREATE TABLE IF NOT EXISTS daily_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    login_time DATETIME,
    logout_time DATETIME,
    avg_stress INTEGER
)
""")

daily_conn.commit()

# =========================
# MEDIAPIPE SETUP
# =========================
base_opts = base_options.BaseOptions(model_asset_path="face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=base_opts,
    running_mode=vision.RunningMode.VIDEO,
num_faces=5)

face_landmarker = FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# =========================
# FUNCTIONS
# =========================
def eye_aspect_ratio(landmarks, eye_ids, w, h):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_ids]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    if C == 0:
        return 0

    return (A + B) / (2.0 * C)

def detect_posture(landmarks, w, h):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    eye_diff = abs((left_eye.y - right_eye.y) * h)
    slouch = nose.z < -0.4

    if eye_diff > 10 and slouch:
        return "Bad (Tilt+Slouch)"
    elif eye_diff > 10:
        return "Bad (Tilted)"
    elif slouch:
        return "Bad (Slouching)"
    return "Good"

def compute_stress(blink_rate, neg_emotion, ear_var, mouse_clicks, key_strokes):
    behavior = (mouse_clicks + key_strokes) / 100

    score = (
        0.3 * blink_rate +
        0.3 * neg_emotion +
        0.2 * ear_var +
        0.2 * behavior
    )
    if np.isnan(score):
        score = 0

    return min(100, int(score * 100))
def reset_tracking():
    global blink_count, ear_history, emotion_score
    global current_emotion, current_posture, current_stress
    global mouse_clicks, key_strokes
    global screen_time, last_activity_time, last_screen_update
    global start_time, last_emotion_time, blink_frames

    blink_count = 0
    ear_history = []
    emotion_score = 0
    current_emotion = "neutral"
    current_posture = "Good"
    current_stress = 0

    mouse_clicks = 0
    key_strokes = 0

    screen_time = 0
    last_activity_time = time.time()
    last_screen_update = time.time()

    start_time = time.time()
    last_emotion_time = time.time()

    blink_frames = 0
# =========================
# GLOBAL VARIABLES
# =========================
stress_history = []
tracking_active = False
cap = cv2.VideoCapture(0)


blink_count = 0
ear_history = []
emotion_score = 0
current_emotion = "neutral"
current_posture = "Good"
current_stress = 0

mouse_clicks = 0
key_strokes = 0

screen_time = 0
last_activity_time = time.time()
last_screen_update = time.time()

start_time = time.time()
last_emotion_time = time.time()

output_frame = None
lock = threading.Lock()
last_saved_time = 0
blink_frames = 0

# =========================
# ACTIVITY TRACKING
# =========================
def on_click(x, y, button, pressed):
    global mouse_clicks, last_activity_time
    if pressed:
        mouse_clicks += 1
        last_activity_time = time.time()

def on_press(key):
    global key_strokes, last_activity_time
    key_strokes += 1
    last_activity_time = time.time()

mouse.Listener(on_click=on_click).start()
keyboard.Listener(on_press=on_press).start()

# =========================
# FLASK
# =========================
app = Flask(__name__)
CORS(app)
app.secret_key = "secret123"
app.register_blueprint(auth)
@app.route("/metrics")
def metrics():
    return jsonify({
    "stress": current_stress,
    "emotion": current_emotion,
    "posture": current_posture,
    "screen_time": int(screen_time),
    "mouse_clicks": mouse_clicks,
    "key_strokes": key_strokes,
    "blinks": blink_count   # ✅ ADD THIS
})

# ✅ OPTIMIZED AI ROUTE
@app.route("/ai_recommendation")
def ai_recommendation():
    if current_stress <= 50:
        return jsonify({"advice": "All good "})

    advice = get_ai_recommendation(
        current_stress,
        current_emotion,
        current_posture,
        int(screen_time)
    )

    return jsonify({"advice": advice})

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

from flask import redirect

@app.route("/dashboard")
def dashboard():
    global tracking_active

    if "user" not in session:
        return redirect("/")

    # ✅ START FRESH TRACKING AFTER LOGIN
    if session.get("tracking_start"):
        session["login_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        reset_tracking()
        tracking_active = True
        session.pop("tracking_start")   # run only once

    return render_template("dashboard.html", name=session["user"])

def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            _, buffer = cv2.imencode(".jpg", output_frame)
            frame = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
from flask import redirect

@app.route("/end_session")
def end_session():
    if "user" not in session:
        return redirect("/")

    name = session["user"]
    email = session["email"]
    login_time = session.get("login_time", "")
    logout_time = time.strftime("%Y-%m-%d %H:%M:%S")
    avg_stress = int(sum(stress_history)/len(stress_history)) if stress_history else 0
    daily_cursor.execute("""
INSERT INTO daily_sessions (name, email, login_time, logout_time, avg_stress)
VALUES (?, ?, ?, ?, ?)
""", (name, email, login_time, logout_time, avg_stress))
    daily_conn.commit()

    user_cursor.execute("""
INSERT INTO user_sessions (name, email, stress)
VALUES (?, ?, ?)
""", (name, email, current_stress))

    user_conn.commit()

    pdf_path = generate_pdf(
        name,
        current_stress,
        blink_count,
        int(screen_time),
        mouse_clicks,
        key_strokes
    )

    send_email(email, pdf_path)
    cursor.execute("""
INSERT INTO reports (email, file_path)
VALUES (?, ?)
""", (email, pdf_path))

    conn.commit()

    # ✅ CLEAR SESSION (LOGOUT)
    global tracking_active
    tracking_active = False
    session.clear()

    # ✅ REDIRECT TO LOGIN
    return render_template("logout.html")
@app.route("/details")
def details():
    if "user" not in session:
        return redirect("/")

    name = session["user"]
    email = session["email"]

    # ✅ FETCH USER HISTORY
    user_cursor.execute("""
    SELECT COUNT(*), AVG(stress)
    FROM user_sessions
    WHERE email = ?
    """, (email,))

    result = user_cursor.fetchone()

    total_sessions = result[0] if result[0] else 0
    avg_stress = int(result[1]) if result[1] else 0
    daily_cursor.execute("""
SELECT login_time, avg_stress
FROM daily_sessions
WHERE email = ?
ORDER BY id DESC
LIMIT 5
""", (email,))
    daily_data = daily_cursor.fetchall()

    return render_template(
        "details.html",
        name=name,
        total_sessions=total_sessions,
        avg_stress=avg_stress,
        blinks=blink_count,
        stress=current_stress,
        daily_data=daily_data   
    )
@app.route("/admin/logs")
def view_logs():
    cursor.execute("SELECT * FROM logins ORDER BY id DESC")
    logins = cursor.fetchall()

    cursor.execute("SELECT * FROM reports ORDER BY id DESC")
    reports = cursor.fetchall()

    return {
        "logins": logins,
        "reports": reports
    }
threading.Thread(target=run_flask, daemon=True).start()

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():
    if not tracking_active:
        time.sleep(0.1)
        continue
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = face_landmarker.detect_for_video(mp_image, int(time.time()*1000))

    if result.face_landmarks:
        landmarks = get_main_face(result.face_landmarks, w, h)

        if landmarks is None:
            continue

        draw_face_box(frame, landmarks, w, h)

        ear = (eye_aspect_ratio(landmarks, LEFT_EYE, w, h) +
               eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)) / 2

        ear_history.append(ear)
        ear = np.mean(ear_history[-3:])
        if np.isnan(ear):
            ear = 0

        EAR_THRESHOLD = 0.2
        CONSEC_FRAMES = 2

        if ear < EAR_THRESHOLD:
            blink_frames += 1
        else:
            if blink_frames >= CONSEC_FRAMES:
                blink_count += 1
            blink_frames = 0

        current_posture = detect_posture(landmarks, w, h)

    # Emotion
    if time.time() - last_emotion_time > 2:
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            current_emotion = analysis[0]['dominant_emotion']
            print("Current Emotion:", current_emotion)
            neg = analysis[0]['emotion'].get('sad', 0) + analysis[0]['emotion'].get('angry', 0)
            emotion_score = neg / 100
            last_emotion_time = time.time()
        except:
            pass

    # Screen time
    now = time.time()
    if now - last_activity_time < 60:
        screen_time += now - last_screen_update
    last_screen_update = now

    # Stress
    elapsed = time.time() - start_time
    blink_rate = blink_count / max(elapsed, 1)
    ear_var = np.var(ear_history[-30:]) if len(ear_history) > 10 else 0
    if np.isnan(ear_var):
        ear_var = 0

    current_stress = compute_stress(
        blink_rate, emotion_score, ear_var, mouse_clicks, key_strokes
    )
    stress_history.append(current_stress)

    # Save DB
    if time.time() - last_saved_time > 10:
        cursor.execute("""
        INSERT INTO sessions (stress, emotion, blinks, screen_time, mouse_clicks, key_strokes)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (current_stress, current_emotion, blink_count, int(screen_time), mouse_clicks, key_strokes))
        conn.commit()
        # 🔥 PRINT LATEST RECORD
        cursor.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()

        print("\n🔥 Latest DB Record:")
        print(row)

        last_saved_time = time.time()

    with lock:
        output_frame = frame.copy()

cap.release()
cv2.destroyAllWindows()
