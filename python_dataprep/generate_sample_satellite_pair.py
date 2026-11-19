"""
Pure Python (zero external dependencies) generator for demo satellite image pairs.
Produces uncompressed 24-bit BMP images readable by JavaFX and standard image viewers.
"""

import os
import struct

def write_bmp(filename, width, height, pixel_grid):
    # BMP rows must be multiple of 4 bytes
    row_padding = (4 - (width * 3) % 4) % 4
    image_size = (width * 3 + row_padding) * height
    file_size = 54 + image_size

    # File Header
    header = bytearray()
    header.extend(b'BM')
    header.extend(struct.pack('<IHHI', file_size, 0, 0, 54))

    # DIB Header (BITMAPINFOHEADER - 40 bytes)
    header.extend(struct.pack('<IiiHHIIiiII',
                              40,          # header size
                              width,
                              -height,     # negative = top-down
                              1,           # planes
                              24,          # 24-bit color
                              0,           # uncompressed (BI_RGB)
                              image_size,
                              2835, 2835,  # 72 DPI
                              0, 0))

    data = bytearray()
    for y in range(height):
        for x in range(width):
            r, g, b = pixel_grid[y][x]
            data.extend(struct.pack('BBB', b, g, r))  # BGR format
        data.extend(b'\x00' * row_padding)

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(data)

def generate_samples():
    # Project root = the folder above python_dataprep/, so this works on any machine.
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Written straight into the app's bundled resources, which the dashboard's "Load Demo Pair" reads.
    output_dir = os.path.join(workspace, "src", "main", "resources", "com", "landwatch", "sample_data")
    os.makedirs(output_dir, exist_ok=True)

    width, height = 400, 300

    # 1. PRE-EVENT: Deep forest canopy with river
    pre_grid = [[(34, 110, 48) for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            shade = ((x * 13 + y * 7) % 35) - 15
            r = max(0, min(255, 34 + shade))
            g = max(0, min(255, 110 + shade))
            b = max(0, min(255, 48 + shade))
            pre_grid[y][x] = (r, g, b)

    # Add river meandering through
    for x in range(width):
        river_y = int(140 + 25 * (x / 80.0) - 10 * ((x % 100) / 50.0))
        for dy in range(-8, 9):
            ry = river_y + dy
            if 0 <= ry < height:
                pre_grid[ry][x] = (25, 75, 155)

    pre_path = os.path.join(output_dir, "siteA_pre.bmp")
    write_bmp(pre_path, width, height, pre_grid)
    print(f"Generated Pre-Event: {pre_path}")

    # 2. POST-EVENT: Deforestation, cleared earth, and settlement encroachment
    post_grid = [row[:] for row in pre_grid]

    # Cleared deforestation polygon (brownish earth)
    for y in range(35, 150):
        for x in range(180, 360):
            # Polygon boundary condition
            if (x - 180) * 0.3 + 35 <= y <= (x - 180) * 0.1 + 145:
                shade = ((x * 19 + y * 11) % 25) - 12
                r = max(0, min(255, 185 + shade))
                g = max(0, min(255, 120 + shade))
                b = max(0, min(255, 75 + shade))
                post_grid[y][x] = (r, g, b)

    # Road from left edge to cleared parcel
    for x in range(0, 200):
        road_y = int(60 + 0.12 * x)
        for dy in range(-2, 3):
            ry = road_y + dy
            if 0 <= ry < height:
                post_grid[ry][x] = (195, 175, 140)

    # Encroachment structures (buildings / sheds)
    for y in range(60, 95):
        for x in range(250, 290):
            if y == 60 or y == 94 or x == 250 or x == 289:
                post_grid[y][x] = (40, 40, 40)
            else:
                post_grid[y][x] = (210, 215, 220)

    for y in range(75, 110):
        for x in range(305, 340):
            if y == 75 or y == 109 or x == 305 or x == 339:
                post_grid[y][x] = (40, 40, 40)
            else:
                post_grid[y][x] = (185, 75, 60)

    post_path = os.path.join(output_dir, "siteA_post.bmp")
    write_bmp(post_path, width, height, post_grid)
    print(f"Generated Post-Event: {post_path}")

if __name__ == "__main__":
    generate_samples()
