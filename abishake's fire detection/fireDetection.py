import cv2         # Library for OpenCV
import threading   # Library for threading
import time        # Library for time functions
import winsound    # Library for system sound (Windows specific)

# Load cascade classifier for fire detection
fire_cascade = cv2.CascadeClassifier('fire_detection_cascade_model.xml') 

# Start video capture from default camera
vid = cv2.VideoCapture(0) 

# Function to play system beep sound
def play_system_beep():
    print("System beep activated!")
    winsound.Beep(1000, 200)  # Example beep sound (1000 Hz frequency, 200 ms duration)

# Variable to keep track of the fire detection rate
fire_count = 0
start_time = time.time()

while True:
    ret, frame = vid.read() # Read video frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5) # Detect fire in the frame

    # Highlight fire with a square and calculate fire coverage percentage
    total_fire_area = 0
    for (x,y,w,h) in fire:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        total_fire_area += w * h

    # Calculate fire coverage percentage
    frame_area = frame.shape[0] * frame.shape[1]
    fire_percentage = (total_fire_area / frame_area) * 100

    # Display fire percentage on frame
    fire_text = "Fire Coverage: {:.2f}%".format(fire_percentage)
    cv2.putText(frame, fire_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display fire percentage in bounding box
    text_width, text_height = cv2.getTextSize(fire_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.rectangle(frame, (20, 20 - text_height - 10), (20 + text_width + 10, 20), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, fire_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display frame
    cv2.imshow('frame', frame)

    # Play system beep if fire is detected
    if len(fire) > 0:
        threading.Thread(target=play_system_beep).start()

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
vid.release()
cv2.destroyAllWindows()
