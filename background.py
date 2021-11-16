import os
import cv2
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")


def plot(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# Mouse callback function
def draw_circle(event, x, y, flags, param):
    global positions
    # If event is Left Button Click then store the coordinate in the lists
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(background_draw, (x, y), 3, (0, 0, 128), -1)
        positions.append([x, y])
        print(positions)


# Reading in images
bg = 'jcdecaux.jpeg'
background_draw = cv2.imread(bg)
background = background_draw.copy()

insert = cv2.imread('hm_in.jpg')
height, width, c = insert.shape

# Corner coordinates for insert
pts = [[0, 0],
       [width, 0],
       [0, height],
       [width, height]]

# Select coordinates for
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
positions = []
count = 0
while True:
    cv2.imshow('image', background_draw)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    if len(positions) == 4:
        break
cv2.destroyAllWindows()
pts, positions = np.array(pts), np.array(positions)

middle_x, middle_y = positions[:, 0].mean(), positions[:, 1].mean()
upper_left, upper_right, lower_left, lower_right = (None, ) * 4
for x, y in positions:
    # upper and left etc.
    if x < middle_x and y < middle_y:
        upper_left = (x, y)
    elif x > middle_x and y < middle_y:
        upper_right = (x, y)
    elif x > middle_x and y > middle_y:
        lower_right = (x, y)
    elif x < middle_x and y > middle_y:
        lower_left = (x, y)

positions = np.array([upper_left, upper_right, lower_left, lower_right])

# Get transformation matrix
h, mask = cv2.findHomography(pts, positions, cv2.RANSAC, 5.0)
height, width, channels = background.shape

# Warp insert and paste into fitting matrix
warped_insert = cv2.warpPerspective(insert, h, (width, height))

# Generate image out of mask, background and insert
# Out of some reason the dimensions/coordinates work differently in this function
background = cv2.fillPoly(background, [positions[[1, 0, 2, 3]]], 0)
out = background + warped_insert
# Plot the results
plot(insert)
plot(warped_insert)
plot(background)
plot(out)
cv2.imwrite("hm_out.png", out)
