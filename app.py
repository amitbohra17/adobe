from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
import svgpathtools
from svgpathtools import svg2paths
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from PIL import Image
import svgwrite


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Function to process SVG paths (regularization)
def process_svg(input_svg_path, output_svg_path):
    paths, attributes = svg2paths(input_svg_path)

    def fit_line(points):
        x = np.array([p.real for p in points])
        y = np.array([p.imag for p in points])
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c

    def fit_circle(points):
        def calc_R(c, x, y):
            return np.sqrt((x - c[0])**2 + (y - c[1])**2)

        def f(c, x, y):
            Ri = calc_R(c, x, y)
            return Ri - Ri.mean()

        x = np.array([p.real for p in points])
        y = np.array([p.imag for p in points])
        center_estimate = (x.mean(), y.mean())
        result = least_squares(f, center_estimate, args=(x, y))
        center = result.x
        radius = calc_R(center, x, y).mean()
        return center, radius

    def is_line(path):
        points = [seg.start for seg in path] + [path[-1].end]
        if len(points) < 2:
            return False
        x = np.array([p.real for p in points])
        y = np.array([p.imag for p in points])
        m, c = fit_line(points)
        y_fit = m * x + c
        return np.allclose(y, y_fit, atol=1)

    def is_circle(path):
        points = [seg.start for seg in path] + [path[-1].end]
        if len(points) < 3:
            return False
        center, radius = fit_circle(points)
        distances = [abs(np.sqrt((p.real - center[0])**2 + (p.imag - center[1])**2) - radius) for p in points]
        return np.allclose(distances, 0, atol=1)

    def regularize_path(path):
        points = [seg.start for seg in path] + [path[-1].end]
        if is_line(path):
            m, c = fit_line(points)
            x_min, x_max = min(p.real for p in points), max(p.real for p in points)
            start = complex(x_min, m * x_min + c)
            end = complex(x_max, m * x_max + c)
            return svgpathtools.Path(svgpathtools.Line(start, end))
        elif is_circle(path):
            center, radius = fit_circle(points)
            start = complex(center[0] + radius, center[1])
            arc1 = svgpathtools.Arc(start=start, radius=radius, rotation=0, large_arc=False, sweep=False, end=complex(center[0] - radius, center[1]))
            arc2 = svgpathtools.Arc(start=arc1.end, radius=radius, rotation=0, large_arc=False, sweep=False, end=start)
            return svgpathtools.Path(arc1, arc2)
        return path

    dwg = svgwrite.Drawing(output_svg_path, profile='tiny')
    for path, attr in zip(paths, attributes):
        regularized_path = regularize_path(path)
        path_data = regularized_path.d()
        dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))
    dwg.save()

# Function to detect and visualize shapes
def detect_shapes(image_path, output_path):
    def detect_shapes(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def complete_shape(image, contour):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            return approx.reshape(-1, 2), "triangle"
        elif len(approx) == 4:
            return approx.reshape(-1, 2), "rectangle"
        elif len(approx) > 4:
            hull = cv2.convexHull(contour)
            hull_approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(hull_approx) == 3:
                return hull_approx.reshape(-1, 2), "triangle"
            else:
                return complete_circle_from_contour(image, contour), "circle"
        return None, None

    def complete_circle_from_contour(image, contour):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        return generate_circle(center, radius)

    def generate_circle(center, radius, num_points=100):
        t = np.linspace(0, 2 * np.pi, num_points)
        x_circle = center[0] + radius * np.cos(t)
        y_circle = center[1] + radius * np.sin(t)
        return np.column_stack((x_circle, y_circle)).astype(np.int32)

    def visualize(image, shape_points, shape_type, output_path):
        image_with_shape = image.copy()
        if shape_type == "circle":
            for i in range(len(shape_points) - 1):
                cv2.line(image_with_shape, tuple(shape_points[i]), tuple(shape_points[i + 1]), (255, 0, 0), 2)
        else:
            cv2.drawContours(image_with_shape, [shape_points], -1, (255, 0, 0), 2)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image_with_shape, cv2.COLOR_BGR2RGB))
        plt.title(f'Completed {shape_type.capitalize()}')
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0, image_with_shape.shape[1])
        plt.ylim(image_with_shape.shape[0], 0)
        plt.savefig(output_path)
        plt.close()

    image = cv2.imread(image_path)
    contours = detect_shapes(image)

    for contour in contours:
        shape_points, shape_type = complete_shape(image, contour)
        if shape_points is not None:
            visualize(image, shape_points, shape_type, output_path)

# Function to detect and visualize symmetry
def detect_symmetry(image_path, output_path):
    def load_image(input_path):
        if input_path.lower().endswith('.csv'):
            image = np.genfromtxt(input_path, delimiter=',')
            if image.max() > 1:
                image = (image > 127).astype(np.uint8) * 255
            return image
        else:
            image = Image.open(input_path).convert('L')
            image = np.array(image)
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary_image

    def is_symmetric(img1, img2):
        if img1.shape != img2.shape:
            return False
        diff = cv2.absdiff(img1, img2)
        return np.sum(diff) == 0

    def detect_and_visualize_symmetry(image_path, output_path):
        image = load_image(image_path)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("No contours found in image")
        contour = max(contours, key=cv2.contourArea)
        contour_mask = np.zeros_like(image)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        h, w = contour_mask.shape
        symmetry_lines = []

        if is_symmetric(contour_mask[:, :w // 2:], np.flip(contour_mask[:, :w // 2:])):
            symmetry_lines.append('horizontal')
        if is_symmetric(contour_mask[:h // 2, :], np.flip(contour_mask[:h // 2:, :])):
            symmetry_lines.append('vertical')
        if is_symmetric(contour_mask, np.transpose(contour_mask)):
            symmetry_lines.append('diagonal_main')
        if is_symmetric(np.flipud(contour_mask), np.transpose(contour_mask)):
            symmetry_lines.append('diagonal_anti')

        plt.imshow(image, cmap='gray')
        for line_type in symmetry_lines:
            if line_type == 'vertical':
                plt.axvline(x=w // 2, color='red', linestyle='--', label='Vertical')
            elif line_type == 'horizontal':
                plt.axhline(y=h // 2, color='blue', linestyle='--', label='Horizontal')
            elif line_type == 'diagonal_main':
                plt.plot([0, w], [0, h], color='green', linestyle='--', label='Diagonal Main')
            elif line_type == 'diagonal_anti':
                plt.plot([0, w], [h, 0], color='yellow', linestyle='--', label='Diagonal Anti')

        plt.legend()
        plt.savefig(output_path)
        plt.close()

    detect_and_visualize_symmetry(image_path, output_path)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_svg', methods=['POST'])
def upload_svg():
    if 'svg_file' not in request.files:
        return "No file part", 400

    svg_file = request.files['svg_file']
    if svg_file.filename == '':
        return "No selected file", 400

    if svg_file:
        input_svg_path = os.path.join(app.config['UPLOAD_FOLDER'], svg_file.filename)
        output_svg_path = os.path.join(app.config['UPLOAD_FOLDER'], 'regularized_' + svg_file.filename)
        svg_file.save(input_svg_path)
        process_svg(input_svg_path, output_svg_path)
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'regularized_' + svg_file.filename)

@app.route('/upload_shape_image', methods=['POST'])
def upload_shape_image():
    if 'shape_image' not in request.files:
        return "No file part", 400

    image_file = request.files['shape_image']
    if image_file.filename == '':
        return "No selected file", 400

    if image_file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + image_file.filename)
        detect_shapes(image_path, output_image_path)

        return send_from_directory(app.config['UPLOAD_FOLDER'], 'processed_' + image_file.filename)

@app.route('/upload_symmetry_image', methods=['POST'])
def upload_symmetry_image():
    if 'symmetry_image' not in request.files:
        return "No file part", 400

    image_file = request.files['symmetry_image']
    if image_file.filename == '':
        return "No selected file", 400

    if image_file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'symmetry_' + image_file.filename)
        detect_symmetry(image_path, output_image_path)

        return send_from_directory(app.config['UPLOAD_FOLDER'], 'symmetry_' + image_file.filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
