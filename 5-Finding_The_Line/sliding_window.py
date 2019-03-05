import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# Load our image
binary_warped = mpimg.imread('warped-example.jpg')

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    lane_width = rightx_base - leftx_base
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    left_fun_x = []
    right_fun_x = []
    left_fun_y = []
    right_fun_y = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low  = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        print(window)
        if window == 0:
            win_xleft_low   = leftx_current - margin
            win_xleft_high  = leftx_current + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)


        # ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        # good_left_inds = binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high].nonzero()
        # print(win_y_low)
        # print(good_left_inds)
        # good_right_inds = binary_warped[win_y_low:win_y_high, win_xright_low:win_xright_high].nonzero()
        # print(good_right_inds)

        high_left = binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high].nonzero()[0] + win_y_low
        high_right = binary_warped[win_y_low:win_y_high, win_xright_low:win_xright_high].nonzero()[0] + win_y_low

        high_left = high_left[-1]
        high_right = high_right[-1]

        leftx_current = binary_warped[high_left, win_xleft_low:win_xleft_high].nonzero()[0] + win_xleft_low
        leftx_current = int(np.mean(leftx_current))

        rightx_current = binary_warped[high_right, win_xright_low:win_xright_high].nonzero()[0] + win_xright_low
        rightx_current = int(np.mean(rightx_current))
        # rightx_current = leftx_current + lane_width

        win_xleft_low   = leftx_current - margin
        win_xleft_high  = leftx_current + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin

        left_fun_x.append(leftx_current)
        left_fun_y.append(high_left)
        right_fun_x.append(rightx_current)
        right_fun_y.append(high_right)

        # leftx_current  = good_left_inds[]
        # rightx_current =
        #
        # win_xleft_low   = leftx_current - margin
        # win_xleft_high  = leftx_current + margin
        # win_xright_low  = rightx_current - margin
        # win_xright_high = rightx_current + margin

        #
        # # Append these indices to the lists
        # left_lane_inds.append(good_left_inds)
        # print(left_lane_inds[0])
        # right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###


    #
    # # Concatenate the arrays of indices (previously was a list of lists of pixels)
    # try:
    #     left_lane_inds = np.concatenate(left_lane_inds)
    #     right_lane_inds = np.concatenate(right_lane_inds)
    # except ValueError:
    #     # Avoids an error if the above is not implemented fully
    #     pass

    # Extract left and right line pixel positions
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds]
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds]

    return out_img, left_fun_x, right_fun_x, left_fun_y, right_fun_y

out_img, left_fun_x, right_fun_x, left_fun_y, right_fun_y = find_lane_pixels(binary_warped)
plt.imshow(out_img)
plt.plot(left_fun_x, left_fun_y, "ro")
plt.plot(right_fun_x, right_fun_y, "bo")
plt.show()

# def fit_polynomial(binary_warped):
#     # Find our lane pixels first
#     leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
#
#     ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
#     left_fit = None
#     right_fit = None
#
#     # Generate x and y values for plotting
#     ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
#     try:
#         left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#         right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#     except TypeError:
#         # Avoids an error if `left` and `right_fit` are still none or incorrect
#         print('The function failed to fit a line!')
#         left_fitx = 1*ploty**2 + 1*ploty
#         right_fitx = 1*ploty**2 + 1*ploty
#
#     ## Visualization ##
#     # Colors in the left and right lane regions
#     out_img[lefty, leftx] = [255, 0, 0]
#     out_img[righty, rightx] = [0, 0, 255]
#
#     # Plots the left and right polynomials on the lane lines
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
#
#     return out_img
#
#
# out_img = fit_polynomial(binary_warped)
#
# plt.imshow(out_img)
