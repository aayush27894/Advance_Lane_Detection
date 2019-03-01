import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

def corners_unwarp(img, nx, ny, mtx, dist):

    #Undistorting using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    #Converting to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    sizex = np.shape(gray)[1]
    sizey = np.shape(gray)[0]

    #Finding the chessboard corners for using as source points
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # defining 4 source points
        src = np.float32([corners[0][0], corners[nx-1][0],
                         corners[-nx][0], corners[-1][0]])

        # defining 4 destination points
        dst = np.float32([[100,100], [sizex-100, 100],
                         [100, sizey-100], [sizex-100, sizey-100]])

        # computing the transform matrix from source and destination points
        M = cv2.getPerspectiveTransform(src, dst)

        # getting the bird's eye view
        warped = cv2.warpPerspective(undist, M, (sizex, sizey))

    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
