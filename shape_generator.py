import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import math
import os

def generate_shape_plot(filename="shape.png", rotate=False):
    """
    Generates an image with a single randomly colored and sized shape (square, rectangle, circle, or triangle)
    on a 50x50 grid. Saves the image and returns the image path and the area of the shape.
    Shapes can be optionally rotated when rotate=True.

    Args:
        filename (str, optional): Name of the output file. Defaults to "shape.png".
        rotate (bool, optional): Whether to apply random rotation to shapes. Defaults to False.
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

    min_shape_dim = 5 
    max_shape_dim = 35 

    if chosen_shape == 'square':
        side = random.uniform(min_shape_dim, max_shape_dim)
        if rotate:
            diagonal = side * math.sqrt(2)
            safe_margin = diagonal / 2
            center_x = random.uniform(min_coord + safe_margin, max_coord - safe_margin)
            center_y = random.uniform(min_coord + safe_margin, max_coord - safe_margin)
            rotation_angle = random.uniform(-45, 45)
            # Define corners relative to center
            half = side / 2
            corners = np.array([
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half]
            ])
            theta = np.radians(rotation_angle)
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rotated_corners = np.dot(corners, rot_matrix) + np.array([center_x, center_y])
            square_patch = patches.Polygon(rotated_corners, closed=True, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(square_patch)
        else:
            x = random.uniform(min_coord, max_coord - side)
            y = random.uniform(min_coord, max_coord - side)
            square = patches.Rectangle((x, y), side, side, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(square)
        area = side * side

    elif chosen_shape == 'rectangle':
        width = random.uniform(min_shape_dim, max_shape_dim)
        height = random.uniform(min_shape_dim, max_shape_dim)
        while abs(width - height) < min_shape_dim / 4:
            height = random.uniform(min_shape_dim, max_shape_dim)
        if rotate:
            diagonal = math.sqrt(width**2 + height**2)
            safe_margin = diagonal / 2
            center_x = random.uniform(min_coord + safe_margin, max_coord - safe_margin)
            center_y = random.uniform(min_coord + safe_margin, max_coord - safe_margin)
            rotation_angle = random.uniform(-45, 45)
            # Define corners relative to center
            half_w = width / 2
            half_h = height / 2
            corners = np.array([
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h]
            ])
            theta = np.radians(rotation_angle)
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rotated_corners = np.dot(corners, rot_matrix) + np.array([center_x, center_y])
            rect_patch = patches.Polygon(rotated_corners, closed=True, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect_patch)
        else:
            x = random.uniform(min_coord, max_coord - width)
            y = random.uniform(min_coord, max_coord - height)
            rectangle = patches.Rectangle((x, y), width, height, facecolor=color, edgecolor='black', linewidth=2)
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
        if rotate:
            # Equilateral triangle for simplicity in margin
            # The farthest vertex from centroid is max(base, height) * sqrt(3)/3 for equilateral, but for general triangle, use half-diagonal
            diagonal = math.sqrt(base**2 + height**2)
            safe_margin = diagonal / 2
            centroid_x = random.uniform(min_coord + safe_margin, max_coord - safe_margin)
            centroid_y = random.uniform(min_coord + safe_margin, max_coord - safe_margin)
            # Points relative to centroid (centered triangle)
            p1 = (-base/2, -height/2)
            p2 = (base/2, -height/2)
            p3 = (0, height/2)
            points = np.array([p1, p2, p3])
            rotation_angle = random.uniform(-45, 45)
            rotation_rad = np.radians(rotation_angle)
            rot_matrix = np.array([[np.cos(rotation_rad), -np.sin(rotation_rad)], [np.sin(rotation_rad), np.cos(rotation_rad)]])
            points = np.dot(points, rot_matrix)
            points = points + np.array([centroid_x, centroid_y])
            triangle_patch = patches.Polygon(points, closed=True, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(triangle_patch)
        else:
            x_start = random.uniform(min_coord, max_coord - base)
            y_start = random.uniform(min_coord, max_coord - height)
            p1 = (x_start, y_start)
            p2 = (x_start + base, y_start)
            p3 = (x_start + base / 2.0, y_start + height)
            points = np.array([p1, p2, p3])
            triangle_patch = patches.Polygon(points, closed=True, facecolor=color, edgecolor='black', linewidth=2)
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
    img_path2, area2 = generate_shape_plot(filename="shape_rotated.png", rotate=True)
    print(f"Generated rotated: {img_path2}, Area: {area2:.2f}")
    img_path1, area1 = generate_shape_plot(filename="shape_area_range.png")
    print(f"Generated: {img_path1}, Area: {area1:.2f}")