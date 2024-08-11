import svgpathtools
import svgwrite
from svgpathtools import svg2paths
import numpy as np
from scipy.optimize import least_squares

# Load the SVG file
input_svg_path = "frag2.svg"
output_svg_path = "regularized_output.svg"

# Parse the SVG paths
paths, attributes = svg2paths(input_svg_path)

def fit_line(points):
    """Fit a straight line to a set of points using linear regression."""
    x = np.array([p.real for p in points])
    y = np.array([p.imag for p in points])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"Line fit: slope={m}, intercept={c}")
    return m, c

def fit_circle(points):
    """Fit a circle to a set of points using least squares."""
    def calc_R(c, x, y):
        return np.sqrt((x - c[0])*2 + (y - c[1])*2)

    def f(c, x, y):
        Ri = calc_R(c, x, y)
        return Ri - Ri.mean()

    x = np.array([p.real for p in points])
    y = np.array([p.imag for p in points])
    x_m = x.mean()
    y_m = y.mean()
    center_estimate = (x_m, y_m)
    result = least_squares(f, center_estimate, args=(x, y))
    center = result.x
    radius = calc_R(center, x, y).mean()
    print(f"Circle fit: center={center}, radius={radius}")
    return center, radius

def is_line(path):
    """Check if a path segment is approximately a straight line."""
    points = [seg.start for seg in path] + [path[-1].end]
    if len(points) < 2:
        return False
    x = np.array([p.real for p in points])
    y = np.array([p.imag for p in points])
    m, c = fit_line(points)
    y_fit = m * x + c
    return np.allclose(y, y_fit, atol=1)

def is_circle(path):
    """Check if a path segment is approximately a circle."""
    points = [seg.start for seg in path] + [path[-1].end]
    if len(points) < 3:
        return False
    center, radius = fit_circle(points)
    distances = [abs(np.sqrt((p.real - center[0])*2 + (p.imag - center[1])*2) - radius) for p in points]
    return np.allclose(distances, 0, atol=1)

def regularize_path(path):
    points = [seg.start for seg in path] + [path[-1].end]
    print(f"Original Path: {path}")
    if is_line(path):
        print("Detected as Line")
        m, c = fit_line(points)
        x_min, x_max = min(p.real for p in points), max(p.real for p in points)
        start = complex(x_min, m * x_min + c)
        end = complex(x_max, m * x_max + c)
        return svgpathtools.Path(svgpathtools.Line(start, end))
    elif is_circle(path):
        print("Detected as Circle")
        center, radius = fit_circle(points)
        radius = complex(radius, radius)  # Cast radius to complex number
        start = complex(center[0] + radius.real, center[1])
        # Create two arcs to form a complete circle
        arc1 = svgpathtools.Arc(
            start=start,
            radius=radius,
            rotation=0,
            large_arc=False,
            sweep=False,
            end=complex(center[0] - radius.real, center[1])
        )
        arc2 = svgpathtools.Arc(
            start=arc1.end,
            radius=radius,
            rotation=0,
            large_arc=False,
            sweep=False,
            end=start
        )
        return svgpathtools.Path(arc1, arc2)
    print("No match found, returning original path")
    return path

# Create a new SVG drawing
dwg = svgwrite.Drawing(output_svg_path, profile='tiny')

# Process each path and add it to the new SVG drawing
for path, attr in zip(paths, attributes):
    regularized_path = regularize_path(path)
    path_data = regularized_path.d()
    dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))

# Save the new SVG file
dwg.save()

print(f"Regularized SVG saved to: {output_svg_path}")