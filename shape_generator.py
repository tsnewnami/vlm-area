import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import math
import os

def generate_shape_plot(filename="shape.png"):
    """
    Generates an image with a single randomly colored and sized shape (square, rectangle, circle, or triangle)
    on a 200x200 grid. Saves the image and returns the image path and the area of the shape.
    Shapes are basic (e.g., no rotation for triangles).

    Args:
        filename (str, optional): Name of the output file. Defaults to "shape.png".
    """
    # Ensure file exists
    output_dir = os.path.dirname(filename)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)
        
    # Image dimensions
    img_width, img_height = 224, 224
    dpi_val = 100

    fig, ax = plt.subplots(1, figsize=(img_width / dpi_val, img_height / dpi_val))
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_aspect('equal', adjustable='box')

    # Grid
    major_ticks = np.arange(0, img_width + 1, 20)
    minor_ticks = np.arange(0, img_width + 1, 5)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='x', labelrotation=90)
    plt.style.use('seaborn-v0_8-whitegrid')

    shapes = ['square', 'rectangle', 'circle', 'triangle']
    chosen_shape = random.choice(shapes)
    
    color = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
    
    area = 0
    
    min_coord = 0
    max_coord = img_width # Assuming square canvas

    # Define shape size constraints
    min_shape_dim = 20  # Minimum dimension (side, diameter, base, height)
    max_shape_dim = 150  # Maximum dimension

    if chosen_shape == 'square':
        side = random.uniform(min_shape_dim, max_shape_dim)
        x = random.uniform(min_coord, max_coord - side)
        y = random.uniform(min_coord, max_coord - side)
        square = patches.Rectangle((x, y), side, side, facecolor=color)
        ax.add_patch(square)
        area = side * side

    elif chosen_shape == 'rectangle':
        width = random.uniform(min_shape_dim, max_shape_dim)
        height = random.uniform(min_shape_dim, max_shape_dim)
        # Ensure width and height are different enough for a rectangle, not a square
        while abs(width - height) < min_shape_dim / 4: # Avoid near-squares, ensure some difference
            height = random.uniform(min_shape_dim, max_shape_dim)

        x = random.uniform(min_coord, max_coord - width)
        y = random.uniform(min_coord, max_coord - height)
        rectangle = patches.Rectangle((x, y), width, height, facecolor=color)
        ax.add_patch(rectangle)
        area = width * height

    elif chosen_shape == 'circle':
        radius = random.uniform(min_shape_dim / 2.0, max_shape_dim / 2.0)
        center_x = random.uniform(min_coord + radius, max_coord - radius)
        center_y = random.uniform(min_coord + radius, max_coord - radius)
        circle_patch = patches.Circle((center_x, center_y), radius, facecolor=color)
        ax.add_patch(circle_patch)
        area = math.pi * (radius ** 2)

    elif chosen_shape == 'triangle':
        base = random.uniform(min_shape_dim, max_shape_dim)
        height = random.uniform(min_shape_dim, max_shape_dim)
        
        x_start = random.uniform(min_coord, max_coord - base)
        y_start = random.uniform(min_coord, max_coord - height)
        
        p1 = (x_start, y_start)
        p2 = (x_start + base, y_start)
        p3 = (x_start + base / 2.0, y_start + height)
        
        points = np.array([p1, p2, p3])
        triangle_patch = patches.Polygon(points, closed=True, facecolor=color)
        ax.add_patch(triangle_patch)
        area = 0.5 * base * height



    fig.set_size_inches(img_width / dpi_val, img_height / dpi_val) 
    plt.savefig(filename, dpi=dpi_val, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    area = "{:.2f}".format(area)
    return filename, area

if __name__ == '__main__':
    img_path1, area1 = generate_shape_plot()
    print(f"Generated: {img_path1}, Area: {area1}")

