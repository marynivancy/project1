import cv2
import mediapipe as mp
import numpy as np

# Load the user image
image_path = 'girl.jpg'
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image {image_path}")
    exit()

# Load accessory image (with alpha channel)
glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)
if glasses is None:
    print("Error: Could not load glasses.png")
    exit()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Convert image to RGB (MediaPipe expects RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process the image and find facial landmarks
results = face_mesh.process(img_rgb)

if not results.multi_face_landmarks:
    print("No face detected!")
    exit()

face_landmarks = results.multi_face_landmarks[0]

h, w, _ = img.shape

# Select landmark points for glasses positioning
left_eye_outer = face_landmarks.landmark[33]
right_eye_outer = face_landmarks.landmark[263]
left_eye_top = face_landmarks.landmark[159]
right_eye_top = face_landmarks.landmark[145]

# Convert normalized coordinates to pixel values
x1 = int(left_eye_outer.x * w)
x2 = int(right_eye_outer.x * w)
y_center = int(min(left_eye_top.y, right_eye_top.y) * h)

# Apply increased scale factor
scale_factor = 1.8  # << Increased from 1.4 to 1.8
eye_width = x2 - x1
glasses_width = int(eye_width * scale_factor)
glasses_height = int(glasses.shape[0] * glasses_width / glasses.shape[1])
glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

# Center position for placing glasses
center_x = (x1 + x2) // 2
roi_x1 = center_x - glasses_width // 2
roi_x2 = roi_x1 + glasses_width
roi_y1 = y_center - int(glasses_height * 0.5)
roi_y2 = roi_y1 + glasses_height

# Boundary checks
roi_x1 = max(0, roi_x1)
roi_y1 = max(0, roi_y1)
roi_x2 = min(w, roi_x2)
roi_y2 = min(h, roi_y2)

# Adjust resized glasses if ROI clipped
glasses_resized = cv2.resize(glasses_resized, (roi_x2 - roi_x1, roi_y2 - roi_y1), interpolation=cv2.INTER_AREA)

# Separate image and alpha channels
glasses_rgb = glasses_resized[:, :, :3]
glasses_alpha = glasses_resized[:, :, 3] / 255.0

# Blend with background
roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
for c in range(3):
    roi[:, :, c] = (glasses_alpha * glasses_rgb[:, :, c] + (1 - glasses_alpha) * roi[:, :, c])

# Place back into original image
img[roi_y1:roi_y2, roi_x1:roi_x2] = roi

# Save and show
output_path = 'rendered_output.jpg'
cv2.imwrite(output_path, img)
print(f"Output saved as {output_path}")

cv2.imshow('Virtual Try-On', img)
cv2.waitKey(0)
cv2.destroyAllWindows()