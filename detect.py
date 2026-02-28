import time
import numpy as np
import cv2
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

MODEL_PATH = "model.tflite"
SCORE_THRESH = 0.5
NUM_THREADS = 4

# --- Load TFLite model ---
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_h = input_details[0]["shape"][1]
in_w = input_details[0]["shape"][2]
in_dtype = input_details[0]["dtype"]

# --- Camera setup ---
picam2 = Picamera2()
# Capture a larger frame for nicer preview; we’ll resize for the model
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(0.5)

def preprocess(frame_rgb):
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    if in_dtype == np.uint8:
        x = resized.astype(np.uint8)
    else:
        x = (resized.astype(np.float32) / 255.0)
    return np.expand_dims(x, axis=0)

def get_outputs():
    # Common TFLite detection outputs:
    # boxes [1,N,4], classes [1,N], scores [1,N], count [1]
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    count = int(interpreter.get_tensor(output_details[3]["index"])[0])
    return boxes, classes, scores, count

print("Press q to quit.")
while True:
    frame = picam2.capture_array()  # RGB888 -> numpy array (H,W,3), RGB order
    # OpenCV uses BGR for display; convert for preview drawing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    x = preprocess(frame)  # model input expects RGB
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    boxes, classes, scores, count = get_outputs()

    h, w = frame_bgr.shape[:2]
    for i in range(count):
        if scores[i] < SCORE_THRESH:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"id:{int(classes[i])} {scores[i]:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("TFLite Detection", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()