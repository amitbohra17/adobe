import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(input_path):
    if input_path.lower().endswith('.csv'):
        # Load from CSV
        image = np.genfromtxt(input_path, delimiter=',')
        if image.max() > 1:
            image = (image > 127).astype(np.uint8) * 255  # Threshold to binary
        return image
    else:
        # Load from image file
        image = Image.open(input_path).convert('L')
        image = np.array(image)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return binary_image

def is_symmetric(img1, img2):
    if img1.shape != img2.shape:
        return False
    diff = cv2.absdiff(img1, img2)
    return np.sum(diff) == 0

def detect_and_visualize_symmetry(image_path):
    # Load the image
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"Image not found or failed to load at path: {image_path}")

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in image")

    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Create an empty mask for drawing the contour
    contour_mask = np.zeros_like(image)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    h, w = contour_mask.shape
    symmetry_lines = []

    # Check for vertical symmetry
    if is_symmetric(contour_mask[:, :w // 2:], np.flip(contour_mask[:, :w // 2:])):
        symmetry_lines.append('horizontal')

    # Check for horizontal symmetry
    if is_symmetric(contour_mask[:h // 2, :], np.flip(contour_mask[:h // 2:, :])):
        symmetry_lines.append('vertical')

    # Check for main diagonal symmetry (top-left to bottom-right)
    if is_symmetric(contour_mask, np.transpose(contour_mask)):
        symmetry_lines.append('diagonal_main')

    # Check for anti-diagonal symmetry (bottom-left to top-right)
    if is_symmetric(np.flipud(contour_mask), np.transpose(contour_mask)):
        symmetry_lines.append('diagonal_anti')

    # Visualize the image and symmetry lines
    plt.imshow(image, cmap='gray')

    for line_type in symmetry_lines:
        if line_type == 'vertical':
            plt.axvline(x=w // 2, color='red', linestyle='--', label='Vertical')
        elif line_type == 'horizontal':
            plt.axhline(y=h // 2, color='blue', linestyle='--', label='Horizontal')
        elif line_type == 'diagonal_main':
            plt.plot([0, w], [0, h], color='green', linestyle='--', label='Diagonal Main')
        elif line_type == 'diagonal_anti':
            plt.plot([0, w], [h, 0], color='purple', linestyle='--', label='Diagonal Anti')

    plt.legend()
    plt.title(f"Symmetry lines: {len(symmetry_lines)}")
    plt.show()

    return symmetry_lines

# Example usage
input_path = 'frag0.png'  # Can be a CSV, PNG, JPG, etc.
symmetry_lines = detect_and_visualize_symmetry(input_path)
print(f"Detected symmetry lines: {symmetry_lines}")