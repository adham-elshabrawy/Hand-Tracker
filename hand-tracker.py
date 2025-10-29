import cv2
import mediapipe as mp
import numpy as np

def calc_angle(pt1, pt2, pt3):
    """
    Calculate the angle of a finger, in mediapipe it tracks three points on each finger
    use this to calculate angle. pt2 is the vertex of the created triangle
    """
    p1 = np.array([pt1.x, pt1.y])
    p2 = np.array([pt2.x, pt2.y])
    p3 = np.array([pt3.x, pt3.y])

    vector1 = p1 - p2
    vector2 = p3 - p2

    #calculate angle using dot product
    cos_angle = np.dot(vector1, vector1) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return np.degrees(angle)

def get_finger_angles(hand_landmarks):
    """
    Calculate bend angles for all fingers, returns dict with finger names & their angles
    """
    landmarks = hand_landmarks.landmark
    angles = {}

    #Thumb uses landmark 1,2,3
    angles['thumb'] = calc_angle(landmarks[1], landmarks[2], landmarks[3])

    #Index uses 5,6,7
    angles['index'] = calc_angle(landmarks[5], landmarks[6], landmarks[7])

    # Middle finger: uses landmarks 9, 10, 11
    angles['middle'] = calc_angle(landmarks[9], landmarks[10], landmarks[11])
    
    # Ring finger: uses landmarks 13, 14, 15
    angles['ring'] = calc_angle(landmarks[13], landmarks[14], landmarks[15])
    
    # Pinky finger: uses landmarks 17, 18, 19
    angles['pinky'] = calc_angle(landmarks[17], landmarks[18], landmarks[19])

    return angles

#initialize hands 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Hand tracker started. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #Calculate finger angles
            angles = get_finger_angles(hand_landmarks)

            #Show angles on screen
            y_offset = 30
            for finger, angle in angles.items():
                text = f"{finger}: {angle:.1f} deg"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
    
    # Display the frame
    cv2.imshow('Hand Tracker', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
