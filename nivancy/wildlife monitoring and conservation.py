import cv2

# Load the image
image = cv2.imread("wild.jpg")  # Ensure the correct file exists

# Check if image loaded
if image is None:
    print("Error: Image not found.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size to detect animals only
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 50000:  # Adjust area range for animals
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Animal Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Animal Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()