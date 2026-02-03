import cv2
import mediapipe as mp
import numpy as np
import socket
import json

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Finger remapping to fix cross-wired servos
FINGER_REMAPPING = {
    'thumb': 'middle',   # Input Thumb -> Output Middle (moves Thumb)
    'middle': 'thumb',   # Input Middle -> Output Thumb (moves Middle)
    'index': 'ring',     # Input Index -> Output Ring (moves Index)
    'ring': 'pinky',     # Input Ring -> Output Pinky (moves Ring)
    'pinky': 'index'     # Input Pinky -> Output Index (moves Pinky)
}

# Fingers that need inverted servo angles (0% -> 0, 100% -> 180)
# 'thumb' and 'middle' here refer to the DETECTED fingers, which map to 
# the Robot Middle and Robot Thumb respectively. Both need inversion.
REVERSE_ANGLES = ['pinky', 'middle']

def get_finger_distance(hand_landmarks, finger_tip, palm_base=0):
    """
    Calculate Euclidean distance between finger tip and palm base.
    Uses normalized coordinates so distance to camera doesn't matter.
    """
    tip = hand_landmarks.landmark[finger_tip]
    base = hand_landmarks.landmark[palm_base]
    
    distance = np.sqrt(
        (tip.x - base.x)**2 + 
        (tip.y - base.y)**2 + 
        (tip.z - base.z)**2
    )
    return distance

def get_all_finger_distances(hand_landmarks):
    """
    Get distances for all 5 fingers.
    Finger tips: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
    """
    distances = {
        'thumb': get_finger_distance(hand_landmarks, 4),
        'index': get_finger_distance(hand_landmarks, 8),
        'middle': get_finger_distance(hand_landmarks, 12),
        'ring': get_finger_distance(hand_landmarks, 16),
        'pinky': get_finger_distance(hand_landmarks, 20)
    }
    return distances

def calibrate(cap, hands, instruction):
    """
    Capture hand state for calibration.
    """
    print(f"\n{instruction}")
    print("Press SPACE when ready...")
    
    calibration_data = None
    
    while True:
        success, frame = cap.read()
        if not success:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display instruction
        cv2.putText(frame, instruction, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press SPACE when ready", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Hand Tracker - Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spacebar
            if results.multi_hand_landmarks:
                calibration_data = get_all_finger_distances(results.multi_hand_landmarks[0])
                print(f"✓ Captured: {calibration_data}")
                break
            else:
                print("No hand detected! Please try again.")
        elif key == ord('q'):
            return None
    
    return calibration_data

def normalize_finger_values(current, open_cal, closed_cal):
    """
    Normalize finger distances to 0-100 scale.
    0 = fully closed, 100 = fully open
    """
    normalized = {}
    for finger in current.keys():
        open_dist = open_cal[finger]
        closed_dist = closed_cal[finger]
        current_dist = current[finger]
        
        # Calculate percentage (0-100)
        if open_dist != closed_dist:
            percentage = ((current_dist - closed_dist) / (open_dist - closed_dist)) * 100
            percentage = np.clip(percentage, 0, 100)  # Clamp between 0-100
        else:
            percentage = 50
        
        normalized[finger] = percentage
    
    return normalized

def send_servo_data(sock, normalized_values):
    """
    Convert normalized finger values to servo angles and send as JSON.
    0% (closed hand) → 180° (closed robotic finger)
    100% (open hand) → 0° (open robotic finger)
    """
    servo_angles = {}
    for finger, percentage in normalized_values.items():
        # Calculate angle based on whether this finger needs inversion
        if finger in REVERSE_ANGLES:
            # 0% -> 0, 100% -> 180
            angle = int(percentage * 1.8)
        else:
            # Standard: 0% -> 180, 100% -> 0
            angle = int((100 - percentage) * 1.8)
            
        angle = int(np.clip(angle, 0, 180))  # Safety clamp & cast to int
        
        if finger in FINGER_REMAPPING:
            target_finger = FINGER_REMAPPING[finger]
            servo_angles[target_finger] = angle
    
    try:
        json_data = json.dumps(servo_angles) + '\n'
        sock.sendall(json_data.encode())
    except Exception as e:
        print(f"Send error: {e}")

# Main program
def main():
    cap = cv2.VideoCapture(0)
    
    print("="*50)
    print("HAND TRACKER - CALIBRATION")
    print("="*50)
    
    # Calibrate open hand
    open_calibration = calibrate(cap, hands, "Show OPEN hand (fingers spread)")
    if open_calibration is None:
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Calibrate closed hand
    closed_calibration = calibrate(cap, hands, "Show CLOSED hand (make a fist)")
    if closed_calibration is None:
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Connect to Pico W
    print("\n" + "="*50)
    print("Connecting to Pico W at 192.168.86.24:5000...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('192.168.0.200', 5000))
        print("✓ Connected to Pico W!")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("Continuing without servo control...")
        sock = None
    
    print("\nCalibration complete! Starting tracking...")
    print("Press 'q' to quit")
    print("="*50 + "\n")
    
    # Main tracking loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get current distances and normalize
                current_distances = get_all_finger_distances(hand_landmarks)
                normalized_values = normalize_finger_values(
                    current_distances, open_calibration, closed_calibration)
                
                # Send to Pico W
                if sock:
                    send_servo_data(sock, normalized_values)
                
                # Display normalized values
                y_offset = 30
                for finger, value in normalized_values.items():
                    text = f"{finger}: {value:.1f}%"
                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 30
        
        cv2.imshow('Hand Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if sock:
        sock.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    