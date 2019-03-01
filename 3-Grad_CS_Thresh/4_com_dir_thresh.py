import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0 , 1, ksize=sobel_kernel)

    abs_sobel = np.abs(sobel)

    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))

    grad_binary = np.zeros_like(gray)

    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return grad_binary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    mag = np.power((np.power(sobelx, 2) + np.power(sobely, 2)), 0.5)

    scaled = np.uint8(255*mag/np.max(mag))

    mag_binary = np.zeros_like(gray)

    mag_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    dir = np.arctan2(sobely, sobelx)

    dir_binary = np.zeros_like(gray)

    dir_binary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1

    return dir_binary

image = mpimg.imread('signs_vehicles_xygrad.png')

ksize = 5

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1,3))

final = np.zeros_like(image[:,:,0])
print(np.shape(final))
final[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) | (dir_binary == 1))] = 1

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(final, cmap='gray')
ax2.set_title('Thresholded', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
