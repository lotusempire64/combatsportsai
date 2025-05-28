from flask import Flask, render_template, request, Response
import os
import cv2
import mediapipe as mp
import openai

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
                frame_landmarks.append({
                    'id': i,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })
            pose_data.append(frame_landmarks)

    cap.release()
    pose.close()

    prompt = (
        f"As a combat sports coach, analyze these training metrics from a pose-estimated video. "
        f"The metrics below were computed using joint movement speeds and spatial positions. "
        f"Note: Some joint data may include unnatural or impossible movements due to tracking glitches. "
        f"Ignore such anomalies and provide feedback based on plausible patterns only.\n\n"
        f"- Number of strikes (wrist speed bursts): {len(movement_patterns['combinations'])}\n"
        f"- Defensive movements (head movement bursts): {stats['defense_score']}\n"
        f"- Footwork activity (ankle motion bursts): {stats['movement_score']}\n"
        f"- Average guard height: {'High' if stats['avg_guard_height'] < 0.4 else 'Medium' if stats['avg_guard_height'] < 0.6 else 'Low'}\n"
        f"- Stance width: {'Wide' if stats['stance_width'] > 0.4 else 'Narrow'}\n"
        f"- Strike speed: {'Fast' if stats['speed_score'] > 0.15 else 'Medium' if stats['speed_score'] > 0.1 else 'Slow'}\n\n"
        f"Based on this data, provide 3 technical strengths and 3 areas for improvement. "
        f"If data seems unreliable or erratic, explain why and suggest how to improve data quality. "
        f"Be direct, specific, and technical in your feedback."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
