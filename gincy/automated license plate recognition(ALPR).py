import cv2
import easyocr
import matplotlib.pyplot as plt

# Reader initialize cheyyuka (English use cheyyunnu)
reader = easyocr.Reader(['en'])

# Vehicle image load cheyyuka
image_path = 'car.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Number plate area OCR cheyyuka
results = reader.readtext(image)

# Result display cheyyuka
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple([int(val) for val in top_left])
    bottom_right = tuple([int(val) for val in bottom_right])
    
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    print("Detected Text:", text)

# Show result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()