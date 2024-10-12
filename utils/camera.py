import cv2
import time

def capture_image(timeout=30):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Check camera connection or permissions.")
        cap.release()
        return None

    print("Press 's' to capture your face. Timeout in 30 seconds...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        cv2.imshow('Capture Image - Press "s" to capture', frame)

        # Capture on 's' key press
        if cv2.waitKey(10) & 0xFF == ord('s'):
            print("Image captured.")
            cap.release()
            cv2.destroyAllWindows()
            return frame

        # Timeout condition
        if time.time() - start_time > timeout:
            print("Timeout: No image captured.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
