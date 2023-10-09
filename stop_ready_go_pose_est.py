import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import serial

### Serial communication with Arduino ###

# Define function for writing to arduino in bytes
arduino = serial.Serial(port="COM3", baudrate=9600, timeout=.1)

def serial_write(x):
    arduino.write(bytes(str(x), "utf-8"))

### MediaPipe setup ###

#set path to pose estimation ML model
model_path = r"C:\Users\ethma\OneDrive\Github\ce23_casa0014_blinking_monster\gesture_recognizer.task"
# Shortened definitions
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
top_gesture = GestureRecognizerResult

# Instantiate hand landmarker with live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if len(result.gestures) > 0:
        gesture = result.gestures[0][0].category_name
        print(gesture)
        if gesture == "Thumb_Up":
            serial_write(1)
        elif gesture == "Thumb_Down":
            serial_write(3)
        else:
            serial_write(0)
    else:
        serial_write(0)

options = GestureRecognizerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_result
)

### Run loop ###

with GestureRecognizer.create_from_options(options) as recognizer:
    # Get livestream from webcam using opencv
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read incorrectly, exit
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display resulting frame while converting it to MediaPipe Image obj
        cv2.imshow("frame", frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognition_result = recognizer.recognize_async(mp_image, timestamp)

        # Press q to quit
        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()