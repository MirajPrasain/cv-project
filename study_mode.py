# This program tracks just the eyes and estimates focus score

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
import pandas as pd


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


# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)

# launching the yolo model 
model = YOLO('yolov5s.pt')


# Timer setup
SESSION_DURATION = 30  # seconds  #takes from the front end.
start_time = time.time()

focus_scores = []
cheat_times = []

# --- Euclidean Distance ---
def euclidean(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

#takes 2 coordinates as parameters and calculates distance. first, vector subtraction inside btrackets,
#and then, norm applies the formula (x2 + y2)^1/2.


# --- Eye Openness (eye_aspect_ratio) ---
def eye_openness(eye_top, eye_bottom, eye_left, eye_right):
    vertical_openness = euclidean(eye_top, eye_bottom)
    horizontal_openness = euclidean(eye_left, eye_right)
    return vertical_openness / horizontal_openness

# --- Iris Position Ratio ---
def iris_position_ratio(iris_center, eye_left, eye_right, eye_top, eye_bottom):
    total_width = euclidean(eye_left, eye_right)
    total_height = euclidean(eye_top, eye_bottom)

    iris_to_left = euclidean(iris_center, eye_left)
    iris_to_top = euclidean(iris_center, eye_top)

    horizontal_ratio = iris_to_left / total_width
    vertical_ratio = iris_to_top / total_height

    return horizontal_ratio, vertical_ratio

def head_tilt_ratio(left_temple, right_temple, nose_tip): 
    left_to_nose = euclidean(left_temple, nose_tip)
    right_to_nose = euclidean(right_temple, nose_tip)

    return left_to_nose / right_to_nose


def head_down_ratio(nose_tip, chin, eye_level): 
    eye_to_nose = euclidean(eye_level, nose_tip)
    chin_to_nose = euclidean(nose_tip, chin)

    return eye_to_nose / chin_to_nose




# Initialize blink_counter outside the function to maintain its value across frames
blink_counter = 0
# --- Focus Score Function ---
def get_focus_score(results, w, h, phone_detected):
    global blink_counter # Declare blink_counter as a global variable to modify it

    if not results.multi_face_landmarks:
        return 0, "No face detected"

    #breaking down the results array to access faces(rows), landmarks (objects inside rows), and x,y values(attributes of landmark object)
    face = results.multi_face_landmarks[0]
    landmarks = face.landmark
    #landmarks are 468 objects each with attribute x,y,z so. landmark[i] accesses the particular landmark out of 466. landmark[n].x accesses the x attribute of the nth landmark

    #gets corresponding landmark from index, and converts it to pixel value
    def get_point(idx):
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)  #returns pixel value for the landmark. w & h are frame height and widths, declared as global variables

    eye_top = get_point(159)  #holds the pixel values/coordinates for 159th landmark, i.e eye top.
    eye_bottom = get_point(145)
    eye_left = get_point(33)
    eye_right = get_point(133)
    iris_center = get_point(468)


    #landmarks for left and right movement
    nose_tip = get_point(1)
    left_temple = get_point(234)
    right_temple = get_point(454)


    #landmark for up and down head movement
    chin = get_point(152)
    
    eye_level = get_point(151)


    eye_aspect_ratio = eye_openness(eye_top, eye_bottom, eye_left, eye_right)
    iris_horizontal, iris_vertical = iris_position_ratio(iris_center, eye_left, eye_right, eye_top, eye_bottom) #ipr function returns two values
    head_tilt_value = head_tilt_ratio(left_temple, right_temple, nose_tip)
    head_down_value = head_down_ratio(nose_tip, chin, eye_level )
    

    focus = 100
    status = "Focused"

    # If looking far away (horizontally or vertically)
    if iris_horizontal < 0.25 or iris_horizontal > 0.75 or iris_vertical < 0.25 or iris_vertical > 0.75:
        focus -= 50


    #checking for horizontal tilts: 
    if head_tilt_value > 1.8: #head turned left
        focus -= 50


    elif head_tilt_value < 0.2: #head turned right
        focus -= 50


    # check for vertical tilts:
    if head_down_value > 1.3:  # head tilted downward
        focus -= 50  # penalize more for looking down
    elif head_down_value < 0.75:  # head tilted upward
        focus -= 50  # mild penalty for looking unnaturally up


    # Slight horizontal or vertical distraction
    if iris_horizontal < 0.4 or iris_horizontal > 0.6:
        focus -= 30

    if iris_vertical < 0.4 or iris_vertical > 0.6:
        focus -= 30


    # Blink = low Eye Aspect Ratio
    if eye_aspect_ratio < 0.2:  #higher value skips frames for blinks
        # The UnboundLocalError occurs because blink_counter is being assigned within this conditional block
        # but might not be assigned if this condition is never met during the function's execution in a specific frame.
        # To fix this, we declare blink_counter as global at the beginning of the function.
        blink_counter += 1
        if blink_counter >= 3:
            return 0, "Eyes Closed"
        # Important: Reset blink_counter if the eye is open again to count consecutive blinks
    elif eye_aspect_ratio >= 0.2:
        blink_counter = 0

    #Phone detected logic 
    if phone_detected: 
        focus = 0
        return max(0, focus), "Phone Detected"

    return max(0, focus), "Focused"  #focus score is 100. i.e max value


# --- Main Loop ---
while True:
    #find the elasped time
    elasped_time = time.time() - start_time 
    if elasped_time > SESSION_DURATION: 
        break


    #Capturing the frame for later processing 
    ret, frame = cap.read()
    if not ret:
        print("Couldn’t capture frame")
        break

    #1.) Logic for Phone detection
    results_yolo = model(frame)[0]  # Get the first (and only) result
    phone_detected = False

    for box in results_yolo.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label == 'cell phone' and conf > 0.5:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    #2.) Processing Frames and calling getfocusscore
    # Flip frame and convert color BEFORE processing with face mesh
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to get face landmarks
    result = face_mesh.process(rgb_frame)  #main line that captures frame, where focus score is called upon 

    # Get frame dimensions AFTER capturing the frame
    h, w, _ = frame.shape

    # Calculate and display focus score AFTER getting the 'result'
    focus_score, status = get_focus_score(result, w, h, phone_detected) #Get focus score is being calculated 



    #3.)Displaying Data and Information on screen 
    cv2.putText(frame, f"Status: {status}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Focus Score: {focus_score}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.putText(frame, f"Status: {status}", (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    timestamp = time.time() - start_time  #getting the timestamps after the session started.

    if status == "Focused": 
        focus_scores.append(focus_score)

    #adding timestamps for frames when focus is less than 50
    if focus_score < 40:
        cheat_times.append(round(timestamp, 2))

    # Drawing landmarks (this should also come AFTER getting 'result')
    left_iris = [468, 469, 470, 471, 472]
    right_iris = [473, 474, 475, 476, 477]
    left_eye_area = [33, 133, 160, 159, 158, 144, 153, 154, 155]
    right_eye_area = [362, 263, 387, 386, 385, 373, 380, 381, 382]
    nose = [1]
    left_temple = [234]
    right_temple = [454]
    eye_level = [151]
    chin = [152]

    if result.multi_face_landmarks:
        if len(result.multi_face_landmarks) > 1:
            status = "Multiple faces detected"

        for face_landmark in result.multi_face_landmarks:
            
            eye_landmarks = left_iris + right_iris + left_eye_area + right_eye_area
            temple_to_nose = nose + left_temple + right_temple
            eye_to_chin= eye_level + nose + chin

            for idx in eye_landmarks:
                lm = face_landmark.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for i in temple_to_nose:
                lm = face_landmark.landmark[i]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x,y), 2, (0,255,0), -1)

            for j in eye_to_chin:
                lm = face_landmark.landmark[j]
                x,y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x,y), 2, (0,255,0), -1 )

    cv2.imshow("Eye tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------
# Display Session Summary Window
# -------------------------------
if focus_scores:
    avg_focus = sum(focus_scores) / len(focus_scores)
    summary_text = f"Session Complete\nAverage Focus: {avg_focus:.2f}\nDuration: {SESSION_DURATION} sec"
else:
    summary_text = "No valid focus scores recorded."

# Create a white image
summary_img = np.ones((300, 500, 3), dtype=np.uint8) * 255

# Draw the text
y0, dy = 80, 50
for i, line in enumerate(summary_text.split('\n')):
    y = y0 + i * dy
    cv2.putText(summary_img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Show the summary window
cv2.imshow("Focus Summary", summary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



def generate_focus_chart(focus_scores, session_duration, filename="focus_trend.png"):
    if not focus_scores:
        print("No focus data to plot.")
        return

    # X-axis: time in seconds (evenly spaced)
    timestamps = [round(i * (session_duration / len(focus_scores)), 2) for i in range(len(focus_scores))]
    #round(value, 2) => rounds to 2 decimal places. for a 30s duration, if 10 focus scores are recorded, x axis should have 30/10  = 3 spacing. i applies the interval for the next time stamp.

    #Y-axis: Smooth focus scores with rolling average (window=10)
    smoothed_scores = pd.Series(focus_scores).rolling(window=10, min_periods=1).mean()

    # Plot setup
    plt.figure(figsize=(10, 5))  #sets up a blank canvas 10inches wide, 5 inches tall
    plt.plot(timestamps, smoothed_scores, color="blue", linewidth=2, label="Smoothed Focus Score")  #draws the line chart

    # Threshold lines
    plt.axhline(50, color='red', linestyle='--', label='Distraction Threshold (50)')
    plt.axhline(80, color='green', linestyle='--', label='High Focus (80)')

    # Labels and legends
    plt.title("Focus Score Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Focus Score")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(filename)
    plt.close()
    print(f"Focus trend chart saved as {filename}")


generate_focus_chart(focus_scores, SESSION_DURATION)


def generate_session_pdf(summary_text, chart_path="focus_trend.png", filename="study_session_report.pdf"):
    # Initialize PDF document
    pdf = FPDF()
    pdf.add_page()  # Add a blank A4 page

    # --- Title Section ---
    pdf.set_font("Arial", "B", 16)  # Set bold font, size 16
    pdf.cell(0, 10, "Focus Session Report", ln=True, align='C')  # Centered title text
    pdf.ln(10)  # Add vertical space

    # --- Timestamp Section ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current date and time
    pdf.set_font("Arial", "", 12)  # Set regular font
    pdf.cell(0, 10, f"Generated on: {timestamp}", ln=True)  # Print timestamp
    pdf.ln(10)  # Add vertical space

    # --- Session Summary Section ---
    pdf.set_font("Arial", "", 12)  # Regular font for summary
    for line in summary_text.split("\n"):  # Split summary text into lines
        pdf.cell(0, 10, line, ln=True)  # Add each line as its own row
    pdf.ln(10)  # Extra spacing before chart

    # --- Chart Section ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Focus Score Chart:", ln=True)  # Chart label
    pdf.image(chart_path, x=10, w=190)  # Add chart image with full width

    # --- Save PDF ---
    pdf.output(filename)  # Write file to disk
    print(f"[✅] PDF report generated: {filename}")  # Confirm to console


summary = (
    "Session Duration: 30 minutes\n"
    "Focus Score: 78%\n"
    "Distractions Detected: 3\n"
    "Phone Alerts: 1\n"
    "Head Turn Events: 2"
)

generate_session_pdf(summary_text=summary)
