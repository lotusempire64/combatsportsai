from flask import Flask, render_template, request
import os
import cv2
import mediapipe as mp
import numpy as np
import openai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# OpenAI API Key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Mediapipe pose initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_speed(current_landmarks, last_landmarks, points):
    total_speed = 0
    for point in points:
        dx = current_landmarks[point].x - last_landmarks[point].x
        dy = current_landmarks[point].y - last_landmarks[point].y
        speed = (dx**2 + dy**2)**0.5
        total_speed += speed
    return total_speed / len(points)

def has_significant_change(current_landmarks, last_landmarks, threshold):
    key_points = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    for point in key_points:
        curr_x = current_landmarks[point].x
        curr_y = current_landmarks[point].y
        last_x = last_landmarks[point].x
        last_y = last_landmarks[point].y
        if abs(curr_x - last_x) > threshold or abs(curr_y - last_y) > threshold:
            return True
    return False

def analyze_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    sample_interval = 15

    last_landmarks = None
    guard_height_samples = []
    stance_width_samples = []
    movement_patterns = {
        'combinations': [],
        'footwork': [],
        'head_movement': [],
    }
    stats = {
        'avg_guard_height': 0,
        'stance_width': 0,
        'movement_score': 0,
        'defense_score': 0,
        'speed_score': 0
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks and frame_count % sample_interval == 0:
            landmarks = results.pose_landmarks.landmark

            if last_landmarks is None or has_significant_change(landmarks, last_landmarks, 0.05):
                if last_landmarks:
                    avg_guard_y = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y +
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y) / 2
                    guard_height_samples.append(avg_guard_y)

                    stance_width = abs(
                        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x -
                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                    stance_width_samples.append(stance_width)

                    wrist_speed = calculate_speed(landmarks, last_landmarks,
                        [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST])
                    if wrist_speed > 0.1:
                        movement_patterns['combinations'].append(wrist_speed)

                    head_movement = calculate_speed(landmarks, last_landmarks,
                        [mp_pose.PoseLandmark.NOSE])
                    if head_movement > 0.05:
                        movement_patterns['head_movement'].append(head_movement)

                    foot_movement = calculate_speed(landmarks, last_landmarks,
                        [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE])
                    if foot_movement > 0.05:
                        movement_patterns['footwork'].append(foot_movement)

                last_landmarks = landmarks

        frame_count += 1
    cap.release()

    if len(guard_height_samples) > 0:
        stats['avg_guard_height'] = sum(guard_height_samples) / len(guard_height_samples)
        stats['stance_width'] = sum(stance_width_samples) / len(stance_width_samples)
        stats['movement_score'] = len(movement_patterns['footwork'])
        stats['speed_score'] = sum(movement_patterns['combinations']) / len(movement_patterns['combinations']) if movement_patterns['combinations'] else 0
        stats['defense_score'] = len(movement_patterns['head_movement'])

    prompt = (
        f"As a combat sports coach, analyze these training metrics:\n"
        f"- Number of strikes: {len(movement_patterns['combinations'])}\n"
        f"- Defensive movements: {stats['defense_score']}\n"
        f"- Footwork activity: {stats['movement_score']}\n"
        f"- Average guard height: {'High' if stats['avg_guard_height'] < 0.4 else 'Medium' if stats['avg_guard_height'] < 0.6 else 'Low'}\n"
        f"- Stance width: {'Wide' if stats['stance_width'] > 0.4 else 'Narrow'}\n"
        f"- Strike speed: {'Fast' if stats['speed_score'] > 0.15 else 'Medium' if stats['speed_score'] > 0.1 else 'Slow'}\n\n"
        f"Provide 3 specific strengths and 3 areas to improve based on these metrics. Be direct and technical in your feedback."
    )
    try:
        response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000,
    )
        return response.choices[0].message.content
    except Exception as e:
        print("Error generating feedback:", e)
        return "Sorry, something went wrong while generating feedback."

    



@app.route("/", methods=["GET", "POST"])
def index():
    feedback = None
    if request.method == "POST":
        file = request.files["video"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            feedback = analyze_video(filepath)
    return render_template("index.html", feedback=feedback)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
