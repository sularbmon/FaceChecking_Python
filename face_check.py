import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN
import imutils

detector = MTCNN()

def is_blurry(image, threshold=100.0, show_debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()

    if show_debug:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.imshow("Original", image)
        cv2.imshow("Blurred", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Laplacian Variance: {variance} — Blurry: {variance < threshold}")
    return variance < threshold, variance

def check_face_position(face, frame_shape):
    x, y, w, h = face
    cx, cy = x + w // 2, y + h // 2
    frame_h, frame_w = frame_shape[:2]
    return (
        0.3 * frame_w < cx < 0.7 * frame_w and
        0.3 * frame_h < cy < 0.7 * frame_h
    )

def estimate_yaw_angle(left_eye, right_eye, nose):
    eye_center = (left_eye + right_eye) / 2
    nose_vector = nose - eye_center
    eye_line = right_eye - left_eye
    yaw_angle = np.degrees(np.arctan2(nose_vector[0], eye_line[0]))
    return yaw_angle

def detect_and_check(image):
    results = detector.detect_faces(image)
    if len(results) == 0:
        return "❌ No face detected"
    if len(results) > 1:
        return "❌ Multiple faces detected"

    face = results[0]
    box = face['box']
    keypoints = face['keypoints']

    # Check if face is centered
    if not check_face_position(box, image.shape):
        return "⚠️ Face is not centered"

    # Check if image is blurry
    blurry, variance = is_blurry(image, threshold=30.0)
    if blurry:
        return "⚠️ Image is blurry"

    # Check resolution (e.g. at least 300x300 for ID)
    if image.shape[0] < 300 or image.shape[1] < 300:
        return "⚠️ Image resolution too low"

    # Check face direction
    left_eye = np.array(keypoints['left_eye'])
    right_eye = np.array(keypoints['right_eye'])
    nose = np.array(keypoints['nose'])

    yaw = estimate_yaw_angle(left_eye, right_eye, nose)
    print(f"Estimated Yaw Angle: {yaw:.2f}")

    if abs(yaw) > 15:  # You can adjust the threshold
        return "❌ Side face detected, only front face is allowed"

    return "✅ Face is valid for ID photo"

# GUI Setup
class FaceCheckerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Photo Checker")
        self.root.geometry("800x800")
        
        self.label_result = Label(root, text="Upload or capture a photo", font=("Arial", 14))
        self.label_result.pack(pady=10)

        self.canvas = Label(root)
        self.canvas.pack()

        Button(root, text="Upload Image", command=self.upload_image).pack(pady=10)
        Button(root, text="Use Webcam", command=self.capture_from_webcam).pack(pady=10)

    def upload_image(self):
        path = filedialog.askopenfilename()
        if path:
            image = cv2.imread(path)
            self.process_image(image)

    def capture_from_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Webcam not availacleable")
            return
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.process_image(frame)

    def process_image(self, image):
        image = imutils.resize(image, width=500)
        result = detect_and_check(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk
        self.label_result.config(text=result)

# Run the app
if __name__ == "__main__":
    root = Tk()
    app = FaceCheckerGUI(root)
    root.mainloop()
