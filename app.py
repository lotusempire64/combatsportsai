from flask import Flask, render_template, request, Response
import os
import cv2
import mediapipe as mp
import openai
import numpy as np
from scipy.ndimage import gaussian_filter1d

client = openai.OpenAI()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)
    frame_count = 0
    pose_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            frame_landmarks = []
            for i, lm in enumerate(results.pose_landmarks.landmark):
                if lm.visibility > 0.6:
                    frame_landmarks.append([lm.x, lm.y, lm.z])
            if frame_landmarks:
                pose_data.append(frame_landmarks)

    cap.release()
    pose.close()

    # Smooth the landmark coordinates
    smoothed_pose_data = []
    for frame in pose_data:
        if len(frame) > 0:
            smoothed = gaussian_filter1d(np.array(frame), sigma=1, axis=0)
            smoothed_pose_data.append(smoothed.tolist())

    # Detect movement based on ankle joint motion
    def calculate_speed(joint_series):
        speeds = []
        for i in range(1, len(joint_series)):
            prev = np.array(joint_series[i - 1])
            curr = np.array(joint_series[i])
            dist = np.linalg.norm(curr - prev)
            speeds.append(dist)
        return speeds

    left_ankle_series = [frame[27] for frame in smoothed_pose_data if len(frame) > 28]  # Left ankle
    right_ankle_series = [frame[28] for frame in smoothed_pose_data if len(frame) > 28] # Right ankle

    footwork_speeds = calculate_speed(left_ankle_series) + calculate_speed(right_ankle_series)
    avg_footwork_speed = np.mean(footwork_speeds) if footwork_speeds else 0

    footwork_detected = avg_footwork_speed > 0.01  # Basic threshold

    prompt = (
        "You are analyzing pose landmarks extracted from a combat sports training video. "
        "Based on the joint positions, movement speeds, and spatial patterns, identify the likely combat sport (e.g., Muay Thai, Boxing, MMA, Kickboxing). "
        "Then provide tailored feedback for that sport.\n\n"
        "IMPORTANT:\n"
        "- Only base your analysis on visible pose movement and positions.\n"
        "- Do NOT mention clinch work, grappling, or submissions unless clearly evident from the joint movement data.\n"
        "- If visibility is poor or the data seems noisy, say so briefly.\n\n"
        f"Footwork activity detected: {'Yes' if footwork_detected else 'No'}\n\n"
        "FEEDBACK FORMAT:\n"
        "Strengths:\n"
        "- Point 1\n"
        "- Point 2\n"
        "- Point 3\n\n"
        "Areas to Improve:\n"
        "- Point 1\n"
        "- Point 2\n"
        "- Point 3\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a combat sports coach and technical movement analyst. Only analyze based on actual pose and motion data. Do not invent events or unseen details."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error generating feedback:", e)
        return "Sorry, something went wrong while generating feedback."

def gen_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    pose.close()

@app.route("/", methods=["GET", "POST"])
def index():
    feedback = None
    video_filename = None
    if request.method == "POST":
        file = request.files["video"]
        if file:
            video_filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            file.save(filepath)
            feedback = analyze_video(filepath)
            return render_template("index.html", feedback=feedback, video_filename=video_filename)

    return render_template("index.html", feedback=feedback, video_filename=video_filename)

@app.route("/video_feed/<filename>")
def video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    return Response(gen_frames(filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

