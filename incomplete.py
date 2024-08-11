import cv2
import numpy as np
import pandas as pd
import svgpathtools
from scipy.interpolate import splprep, splev
from svgpathtools import svg2paths2

def load_input(input_type, input_data):
    if input_type == 'csv':
        return load_from_csv(input_data)
    elif input_type == 'svg':
        return load_from_svg(input_data)
    elif input_type == 'doodle':
        return load_from_doodle(input_data)
    elif input_type == 'shape':
        return load_from_shape(input_data)
    elif input_type == 'image':
        return load_from_image(input_data)
    else:
        raise ValueError("Unsupported input type")

def load_from_csv(file_path):
    # Assume CSV contains two columns: 'x' and 'y'
    data = pd.read_csv(file_path)
    points = data[['x', 'y']].values
    return [points]  # Return as a list of points

def load_from_svg(file_path):
    paths, _ = svg2paths2(file_path)
    contours = []
    for path in paths:
        points = []
        for line in path:
            points.extend(np.array([line.start.real, line.start.imag]))
        contours.append(np.array(points).reshape(-1, 2))
    return contours

def load_from_doodle(image_path):
    return load_from_image(image_path)

def load_from_shape(shape_description):
    # Assume shape_description is a dict like {'type': 'circle', 'center': (x, y), 'radius': r}
    if shape_description['type'] == 'circle':
        center = shape_description['center']
        radius = shape_description['radius']
        angles = np.linspace(0, 2 * np.pi, 100)
        points = np.array([(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles])
        return [points]
    else:
        raise ValueError("Unsupported shape type")

def load_from_image(image_path):
    image = cv2.imread(image_path)
    edges = detect_edges(image)
    contours = find_contours(edges)
    return contours

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [np.squeeze(contour) for contour in contours]

def complete_curve(contour):
    diff = np.diff(contour, axis=0)
    distances = np.sqrt((diff**2).sum(axis=1))
    threshold = 10  # You may adjust this threshold
    gaps = np.where(distances > threshold)[0]

    if len(gaps) == 0:
        return contour

    contour1 = contour[:gaps[0] + 1]
    contour2 = contour[gaps[0] + 1:]

    tck, u = splprep([contour1[:, 0], contour1[:, 1]], s=0)
    new_points = splev(np.linspace(0, 1, len(contour2)), tck)

    completed_contour = np.vstack([contour1, np.array(new_points).T, contour2])

    return completed_contour

def draw_curve(image, contour):
    for i in range(len(contour) - 1):
        cv2.line(image, tuple(contour[i]), tuple(contour[i + 1]), (0, 255, 0), 2)

def process_input(input_type, input_data):
    contours = load_input(input_type, input_data)
    completed_contours = []
    for contour in contours:
        completed_contour = complete_curve(contour)
        completed_contours.append(completed_contour)
    return completed_contours

# Example usage:

# For CSV input:
# completed_contours = process_input('csv', 'input_file.csv')

# For SVG input:
# completed_contours = process_input('svg', 'input_file.svg')
completed_contours = process_input('svg','problems/occlusion1.svg')
# For doodle input:
# completed_contours = process_input('doodle', 'doodle_image.png')

# For regular shape input:
# shape_desc = {'type': 'circle', 'center': (50, 50), 'radius': 30}
# completed_contours = process_input('shape', shape_desc)

# For image input:
# completed_contours = process_input('image', 'input_image.png')

# Drawing the result (for images and doodles):
# output_image = np.zeros((500, 500, 3), dtype=np.uint8)
# for contour in completed_contours:
#     draw_curve(output_image, contour)

# cv2.imshow('Completed Curves', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()