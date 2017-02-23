#!/usr/bin/env python

import sys
from PIL import Image

image = Image.open(sys.argv[1])
modified_image = Image.open(sys.argv[2])

# image_list = list(image.getdata())
# modified_image_list = list(modified_image.getdata())

# print(image_list)

width, height = image.size

result_image = Image.new("RGBA", (width, height))
for i in range(width):
    for j in range(height):
        new_pixel = modified_image.getpixel((i, j)) if modified_image.getpixel((i, j)) != image.getpixel((i, j)) else (0, 0, 0, 0)
        result_image.putpixel((i, j), new_pixel)
result_image.save("ans_two.png")
