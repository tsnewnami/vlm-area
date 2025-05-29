import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import math
import os

def generate_shape_plot(output_dir_override=None):
    """
    Generates an image with a single randomly colored and sized shape (square, rectangle, circle, or triangle)
    on a 200x200 grid. Saves the image and returns the image path and the area of the shape.
    Shapes are basic (e.g., no rotation for triangles).

    Args:
        output_dir_override (str, optional): If provided, overrides the default output directory.
                                             Defaults to None, which means images are saved in "data".
    """
    # Image dimensions
    img_width, img_height = 200, 200
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
    max_shape_dim = 80  # Maximum dimension

    if chosen_shape == 'square':
        side = random.uniform(min_shape_dim, max_shape_dim)
        x = random.uniform(min_coord, max_coord - side)
        y = random.uniform(min_coord, max_coord - side)
        square = patches.Rectangle((x, y), side, side, facecolor=color)
        ax.add_patch(square)
        area = side * side
        print(f"Generated Square: (x={x:.2f}, y={y:.2f}), side={side:.2f}, area={area:.2f}, color={color}")

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
        print(f"Generated Rectangle: (x={x:.2f}, y={y:.2f}), width={width:.2f}, height={height:.2f}, area={area:.2f}, color={color}")

    elif chosen_shape == 'circle':
        # Diameter will be between min_shape_dim and max_shape_dim
        radius = random.uniform(min_shape_dim / 2.0, max_shape_dim / 2.0)
        # Ensure center allows circle to be fully within bounds
        center_x = random.uniform(min_coord + radius, max_coord - radius)
        center_y = random.uniform(min_coord + radius, max_coord - radius)
        circle_patch = patches.Circle((center_x, center_y), radius, facecolor=color) # Renamed to avoid conflict
        ax.add_patch(circle_patch)
        area = math.pi * (radius ** 2)
        print(f"Generated Circle: center=({center_x:.2f}, {center_y:.2f}), radius={radius:.2f}, area={area:.2f}, color={color}")

    elif chosen_shape == 'triangle':
        # Generate an isosceles triangle with a horizontal base
        base = random.uniform(min_shape_dim, max_shape_dim)
        height = random.uniform(min_shape_dim, max_shape_dim)
        
        x_start = random.uniform(min_coord, max_coord - base)
        y_start = random.uniform(min_coord, max_coord - height)
        
        p1 = (x_start, y_start)
        p2 = (x_start + base, y_start)
        p3 = (x_start + base / 2.0, y_start + height)
        
        points = np.array([p1, p2, p3])
        triangle_patch = patches.Polygon(points, closed=True, facecolor=color) # Renamed
        ax.add_patch(triangle_patch)
        area = 0.5 * base * height
        print(f"Generated Triangle: p1={p1}, p2={p2}, p3={p3}, base={base:.2f}, height={height:.2f} area={area:.2f}, color={color}")

    # Create directory for images if it doesn't exist
    if output_dir_override:
        output_dir = output_dir_override
    else:
        output_dir = "data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img_filename = f"shape_{random.randint(1000,9999)}.png" # Changed filename
    img_path = os.path.join(output_dir, img_filename)

    fig.set_size_inches(img_width / dpi_val, img_height / dpi_val) 
    plt.savefig(img_path, dpi=dpi_val, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    area = round(area, 2)
    print(f"Saved plot to {img_path}")
    return img_path, area

if __name__ == '__main__':
    print("Generating a test shape in the 'data' directory...")
    img_path, area = generate_shape_plot(output_dir_override="data")
    print(f"Test shape generated: Image saved at {img_path}, Area: {area}\\n")

    # print("Generating multiple shapes in 'generated_shapes' directory for general testing:")
    # for i in range(3): 
    #     img_path_gen, area_gen = generate_shape_plot()
    #     print(f"Iteration {i+1}: Image saved at {img_path_gen}, Area: {area_gen:.2f}\\n")
