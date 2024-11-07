import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = "extracted_frame.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the region of interest (ROI)
x_start, x_end = 200, 300
roi = image[:, x_start:x_end]  # Full height, columns 200 to 250

# Display the original image and the ROI
plt.figure(figsize=(10, 5))

# Show original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Show ROI
plt.subplot(1, 2, 2)
plt.imshow(roi, cmap='gray')
plt.title("Region of Interest (200-250)")
plt.axis('off')

plt.show()
