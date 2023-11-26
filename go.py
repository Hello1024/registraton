
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle, vortex
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.exposure import rescale_intensity
import statistics

import cv2
from skimage import img_as_float
import numpy as np

# Open the webcam (usually the first device in the list, hence 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Capture a single frame
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()

# --- Display
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 4))

quiv=0

while True:
    ret, frame = cap.read()

    # If we got frames, show them
    if ret:
        # Convert from OpenCV BGR format to RGB format as expected by skimage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = frame_rgb[600:1000, 400:800]
        frame_rgb = cv2.resize(frame_rgb, (50, 50))
        
        # Convert the image into skimage format (floats instead of ints)
        frame_skimage = img_as_float(frame_rgb)


    image0, image1 = vortex()

    image0 = rescale_intensity(image0[0:50,0:50])
    frame_skimage = rescale_intensity(frame_skimage)

    # --- Compute the optical flow
    v, u = optical_flow_ilk(image0, frame_skimage, radius=20)

    # --- Compute flow magnitude
    norm = np.sqrt(u ** 2 + v ** 2)

    
    # --- Sequence image sample

    ax0.imshow(frame_skimage, cmap='gray')
    ax0.set_title("Sequence image sample")
    ax0.set_axis_off()

    ax1.imshow(image0, cmap='gray')
    ax1.set_title("Sequence image sample")
    ax1.set_axis_off()

    # --- Quiver plot arguments

    nvec = 20  # Number of vectors to be displayed along each image dimension
    nl, nc = image0.shape
    step = max(nl//nvec, nc//nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]

    print(u[25,25], v[25,25])

    ax2.imshow(norm)

    if quiv:
        quiv.remove()
    quiv = ax2.quiver(x, y, u_, v_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)
    ax2.set_title("Optical flow magnitude and vector field")
    ax2.set_axis_off()
    fig.tight_layout()

    #plt.show()
    plt.draw()
    plt.pause(0.001)