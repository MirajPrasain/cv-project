# EXAM MODE: Counts discrete events of cheating behaviors
# (phone checks, multiple faces, extreme head turns, head down)

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import pandas as pd 
import matplotlib.pyplot as plt  # import matplotlib as plt is incorrect



from fpdf import FPDF
from datetime import datetime


# --- Initialize Mediapipe ---
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Initialize Webcam & YOLO for phone detection ---
cap = cv2.VideoCapture(0)
model = YOLO('yolov5s.pt')

# --- Session timer ---
SESSION_DURATION = 30  # seconds
start_time = time.time()

cheat_times = [] 
cheat_event = []

# --- Utility for distance calculations ---
def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# --- Head pose heuristics ---
def head_tilt_ratio(left_temple, right_temple, nose_tip):
    return euclidean(left_temple, nose_tip) / euclidean(right_temple, nose_tip)

def head_down_ratio(nose_tip, chin, eye_level):
    return euclidean(eye_level, nose_tip) / euclidean(nose_tip, chin)

# --- Event counters & state flags ---
phone_checks   = 0
phone_flag     = False

face_events    = 0
face_flag      = False

turn_events    = 0
turn_flag      = False

down_events    = 0
down_flag      = False

# --- Main Loop ---
while True:
    elapsed = time.time() - start_time
    if elapsed > SESSION_DURATION:
        break

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 1) PHONE DETECTION
    results_yolo = model(frame)[0]
    phone_detected = False
    for box in results_yolo.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        if label == 'cell phone' and conf > 0.5:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "PHONE!", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            break

    # PHONE event logic
    if phone_detected and not phone_flag:
        phone_checks += 1
        phone_flag = True
        cheat_times.append(time.time() - start_time)
        cheat_event.append(4)
    elif not phone_detected:
        phone_flag = False

    # 2) FACE MESH PROCESSING
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    h, w, _ = frame.shape

    # MULTIPLE FACES event logic
    multi_face = bool(result.multi_face_landmarks and
                      len(result.multi_face_landmarks) > 1)
    if multi_face and not face_flag:
        face_events += 1
        face_flag = True
        cheat_times.append(time.time() - start_time)
        cheat_event.append(3)
    elif not multi_face:
        face_flag = False

    # If no face at all, skip pose checks
    if not result.multi_face_landmarks:
        cv2.putText(frame, "NO FACE", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Exam Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # SINGLE FACE: compute pose
    face = result.multi_face_landmarks[0]
    lm   = face.landmark
    def P(i): return (int(lm[i].x * w), int(lm[i].y * h))

    nose        = P(1)
    left_temple = P(234)
    right_temple= P(454)
    chin        = P(152)
    eye_lvl     = P(151)

    tilt = head_tilt_ratio(left_temple, right_temple, nose)
    down = head_down_ratio(nose, chin, eye_lvl)

    # EXTREME TURN event logic
    extreme_turn = (tilt > 1.5) or (tilt < 0.67)
    if extreme_turn and not turn_flag:
        turn_events += 1
        turn_flag = True
        cheat_times.append(time.time() - start_time)
        cheat_event.append(2)
    elif not extreme_turn:
        turn_flag = False

    # HEAD DOWN event logic
    looking_down = down > 1.4
    if looking_down and not down_flag:
        down_events += 1
        down_flag = True
        cheat_times.append(time.time() - start_time)
        cheat_event.append(1)
    elif not looking_down:
        down_flag = False

    # Display warnings
    if multi_face:
        cv2.putText(frame, "MULTIPLE FACES!", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if extreme_turn:
        cv2.putText(frame, "LOOKING SIDEWAYS!", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if looking_down:
        cv2.putText(frame, "LOOKING DOWN!", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Exam Mode", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# --- Session Summary ---
lines = [
    "Exam Complete",
    f"Phone checks:           {phone_checks}",
    f"Multiple-face events:   {face_events}",
    f"Extreme turn events:    {turn_events}",
    f"Head-down events:       {down_events}"
]

# Render summary window
summary_h = 200 + 30 * len(lines)
summary_img = np.ones((summary_h, 400, 3), dtype=np.uint8) * 255
for i, line in enumerate(lines):
    y = 40 + i * 30
    cv2.putText(summary_img, line, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

cv2.imshow("Exam Summary", summary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def generate_cheat_chart(cheat_times, cheat_event, session_duration, filename = "cheat_status.png"): 
    if not cheat_times: 
        print("No cheat data to plot") 
        return 
    
    #X-axis -> the time stamp for cheating event to occur
    cheat_times = [round(i * (session_duration/len(cheat_times)),2) for i in range(len(cheat_times))] 

    #Y axis -> the cheating event that occured. Direclty use the cheating event array when calling the function 


    plt.figure(figsize = (10, 5))
    plt.plot(cheat_times, cheat_event, color = 'red', linestyle = 'None', marker = 'o', markersize = 8)


    plt.title("Cheating Event and Corresponding Timestamp") 
    plt.xlabel("Time(seconds)")
    plt.ylabel("Cheating Event Index")
    plt.ylim(4)

    plt.grid(True) 
    plt.legend() 
    plt.tight_layout(pad = 3.0) 
    plt.yticks([1, 2, 3, 4], [
    "Looking Down",
    "Extreme Turn",
    "Multiple Faces",
    "Phone Detected"
    ])
    

    #Save Plot 
    plt.savefig(filename, bbox_inches = 'tight') 
    plt.close()
    print(f"Cheat chart saved as {filename}")




generate_cheat_chart(cheat_times, cheat_event, SESSION_DURATION)


def generate_session_pdf(summary_text, chart_path="cheat_status.png", filename="exam_session_report.pdf"):
    # Initialize PDF document
    pdf = FPDF()
    pdf.add_page()  # Add a blank A4 page

    # --- Title Section ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Exam Session Report", ln=True, align='C')
    pdf.ln(10)

    # --- Timestamp Section ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {timestamp}", ln=True)
    pdf.ln(10)

    # --- Summary Section ---
    pdf.set_font("Arial", "", 12)
    for line in summary_text.split("\n"):
        pdf.cell(0, 10, line, ln=True)
    pdf.ln(10)

    # --- Cheat Chart Image ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Cheating Events Timeline:", ln=True)
    pdf.image(chart_path, x=10, w=190)

    # --- Save PDF File ---
    pdf.output(filename)
    print(f"[✅] PDF report generated: {filename}")

summary = (
    "Session Duration: 30 seconds\n"
    "Phone Checks: 1\n"
    "Multiple Faces Detected: 2\n"
    "Extreme Head Turns: 3\n"
    "Head Down Instances: 2"
)

generate_session_pdf(summary_text=summary)
