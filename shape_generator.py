import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import math
import os
from matplotlib.transforms import Affine2D

def generate_shape_plot(filename="shape.png"):
    """
    Generates an image with a single randomly colored and sized shape (square, rectangle, circle, or triangle)
    on a 50x50 grid. Saves the image and returns the image path and the area of the shape.
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
    grid_size = 50  # New grid size
    dpi_val = 100

    fig, ax = plt.subplots(1, figsize=(img_width / dpi_val, img_height / dpi_val))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal', adjustable='box')

    # Grid - simplified and bolder
    major_ticks = np.arange(0, grid_size + 1, 10)  # Every 10 units
    minor_ticks = np.arange(0, grid_size + 1, 5)   # Every 5 units
    
    # Set up major grid lines
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid(which='major', color='black', linestyle='-', linewidth=1.0, alpha=0.7)
    
    # Add minor ticks without grid lines
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(which='minor', length=4, color='black', width=1)
    ax.tick_params(which='major', length=7, color='black', width=1.5)
    
    ax.tick_params(axis='x', labelrotation=90)
    plt.style.use('seaborn-v0_8-whitegrid')

    shapes = ['square', 'rectangle', 'circle', 'triangle']
    chosen_shape = random.choice(shapes)
    
    color = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
    
    area = 0
    
    min_coord = 0
    max_coord = grid_size  # Using new grid size

    min_shape_dim = 8   
    max_shape_dim = 35 

    if chosen_shape == 'square':
        side = random.uniform(min_shape_dim, max_shape_dim)
        # Compute half-diagonal
        half_diag = (side * math.sqrt(2)) / 2
        center_x = random.uniform(min_coord + half_diag, max_coord - half_diag)
        center_y = random.uniform(min_coord + half_diag, max_coord - half_diag)
        x = center_x - side / 2
        y = center_y - side / 2
        angle = random.uniform(0, 360)
        square = patches.Rectangle((x, y), side, side, facecolor=color, edgecolor='black', linewidth=2)
        t = Affine2D().rotate_deg_around(center_x, center_y, angle) + ax.transData
        square.set_transform(t)
        ax.add_patch(square)
        area = side * side

    elif chosen_shape == 'rectangle':
        width = random.uniform(min_shape_dim, max_shape_dim)
        height = random.uniform(min_shape_dim, max_shape_dim)
        while abs(width - height) < min_shape_dim / 4:
            height = random.uniform(min_shape_dim, max_shape_dim)
        # Compute half-diagonal
        half_diag = (math.sqrt(width ** 2 + height ** 2)) / 2
        center_x = random.uniform(min_coord + half_diag, max_coord - half_diag)
        center_y = random.uniform(min_coord + half_diag, max_coord - half_diag)
        x = center_x - width / 2
        y = center_y - height / 2
        angle = random.uniform(0, 360)
        rectangle = patches.Rectangle((x, y), width, height, facecolor=color, edgecolor='black', linewidth=2)
        t = Affine2D().rotate_deg_around(center_x, center_y, angle) + ax.transData
        rectangle.set_transform(t)
        ax.add_patch(rectangle)
        area = width * height

    elif chosen_shape == 'circle':
        radius = random.uniform(min_shape_dim / 2.0, max_shape_dim / 2.0)
        center_x = random.uniform(min_coord + radius, max_coord - radius)
        center_y = random.uniform(min_coord + radius, max_coord - radius)
        circle_patch = patches.Circle((center_x, center_y), radius, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle_patch)
        area = math.pi * (radius ** 2)

    elif chosen_shape == 'triangle':
        base = random.uniform(min_shape_dim, max_shape_dim)
        height = random.uniform(min_shape_dim, max_shape_dim)
        # Initial triangle points (centered at origin for easier rotation and placement)
        p1 = np.array([-base / 2, -height / 3])
        p2 = np.array([base / 2, -height / 3])
        p3 = np.array([0, 2 * height / 3])
        points = np.array([p1, p2, p3])
        # Centroid is at (0,0) by construction
        max_dist = np.linalg.norm(points, axis=1).max()
        centroid_x = random.uniform(min_coord + max_dist, max_coord - max_dist)
        centroid_y = random.uniform(min_coord + max_dist, max_coord - max_dist)
        angle = random.uniform(0, 360)
        theta = np.deg2rad(angle)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated_points = (points @ rot_matrix.T) + np.array([centroid_x, centroid_y])
        triangle_patch = patches.Polygon(rotated_points, closed=True, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(triangle_patch)
        area = 0.5 * base * height



    fig.set_size_inches(img_width / dpi_val, img_height / dpi_val) 
    plt.savefig(filename, dpi=dpi_val, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    area = round(float(area), 2)
    return filename, area

if __name__ == '__main__':
    img_path1, area1 = generate_shape_plot()
    print(f"Generated: {img_path1}, Area: {area1:.2f}")

