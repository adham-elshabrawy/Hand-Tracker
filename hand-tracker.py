import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
    
    print("\n" + "="*50)
    print("Calibration complete! Starting tracking...")
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
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()