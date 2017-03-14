import numpy as np
import glob
import cv2
import pickle
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LineTracker import LineTracker
from moviepy.editor import VideoFileClip


def get_points(folder_name, out_folder, corner_count, force=False):
    """

    :param folder_name:
    :param out_folder:
    :param corner_count:
    :param force:
    :return:
    """

    output_pickle = 'corners_pickle.p'

    if not os.path.exists(output_pickle) or force:
        # Make a list of calibration images
        # Glob is useful because there is a pattern in the image file names
        file_name_pattern = folder_name + '/*.jpg'
        images = glob.glob(file_name_pattern)

        # Initialize the arrays to store the corner information
        index_array = []  # this is a 3D array with x, y, z grid locations (real world space)
        corners_array = []  # this array will store the corner points in image plane

        # Each chess board has 9x6 corners to detect (inside corners)
        # prepare indices, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # top left corner will be (0,0,0) and bottom right corner (8,6,0)
        indices = np.zeros((corner_count[0] * corner_count[1], 3), np.float32)

        # Now we can use numpy's mgrid to populate the content of the indices array
        # We will only assign values to the x, y coordinates
        # z position will always be zero as images are 2D
        indices[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        for idx, img_name in enumerate(images):

            print("Working on calibration image # ", idx+1)

            # Read image using cv2
            img = cv2.imread(img_name)
            # Convert the colorspace to grayscale - reading images using cv2 returns BGR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners using cv2's findChessboardCorners function
            # This function requires and image and expected number of inside corners in x and y directions
            # The last argument to the function is any flag that we might have. At this points we will set it to None
            # Last parameter is for any flags - we don't have any
            # The function returns whether or not it was successful and the corner locations
            ret, corners = cv2.findChessboardCorners(gray, corner_count, None)

            # When we can find the corners, we will add the resulting information to our arrays
            if ret:

                # indices will not change
                index_array.append(indices)
                # Add corners for each image that is successfully identified
                corners_array.append(corners)

                # Draw and show the corners on each image using cv2's function
                cv2.drawChessboardCorners(img, corner_count, corners, ret)
                # cv2.imshow('calibration_image', img)
                # cv2.waitKey(10)
                cv2.imwrite(os.path.join(out_folder, str('calibration_points_' + str(idx+1) + '.jpg')), img)

        cv2.destroyAllWindows()

        # Pickle the results for later use
        corners_pickle = dict()
        corners_pickle["corners"] = corners_array
        corners_pickle["indices"] = index_array
        pickle.dump(corners_pickle, open(output_pickle, "wb"))

    else:
        print('--corners_pickle.p-- file already exist! Use \'force=True\' to overwrite')


def get_calibration(out_folder, test_folder, test_img, force=False):
    """

    :param out_folder:
    :param test_folder:
    :param test_img:
    :param force:
    :return:
    """

    input_pickle = 'corners_pickle.p'
    output_pickle = 'calibration.p'

    if not os.path.exists(output_pickle) or force:

        if os.path.exists(input_pickle):

            # Read the corner information from the pickle file
            corners_pickle = pickle.load(open(input_pickle, 'rb'))
            indices = corners_pickle['indices']
            corners = corners_pickle['corners']

            # Read the calibration test image
            img = cv2.imread(os.path.join(test_folder, test_img))
            img_size = (img.shape[1], img.shape[0])

            # Do camera calibration given object points and image points
            # mtx is the camera matrix
            # dist = distortion coefficients
            # rvecs, tvecs = position of the camera in real world with rotation and translation vecs
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(indices, corners, img_size, None, None)

            # Test calibration on an image - undistort and save
            # This is usually called destination image
            dst = cv2.undistort(img, mtx, dist, None, mtx)
            cv2.imwrite(os.path.join(out_folder, test_img), dst)

            # Save calibration matrix and distortion coefficients
            calibration = dict()
            calibration["mtx"] = mtx
            calibration["dist"] = dist
            pickle.dump(calibration, open(output_pickle, "wb"))

            print('--Calibration.p-- saved! ')

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=15)
            ax2.imshow(dst)
            ax2.set_title('Undistorted Image', fontsize=15)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            img_name = os.path.splitext(test_img)[0]
            plt.savefig(os.path.join(out_folder, str(img_name + '_compare')))
            plt.close()

        else:
            sys.exit('--corners_pickle.p-- does not exist! Call `get_points()` function first')
    else:
        print('--calibration.p-- file already exist! Use \'force=True\' to overwrite')


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Function that applies Sobel in x or y with a given kernel size
    Takes the absolute value of the gradient
    Scales to 8bit and returns a mask after checking values with the threshold range
    :param img:
    :param orient:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient.upper() == 'X':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient.upper() == 'Y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        sys.exit('Orientation should be a member of (x, y) in abs_sobel_thresh function')

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """
    Calculates the derivatives in x and y directions with a given kernel size
    Uses the resultant magnitude of the derivatives to mask the image
    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    scale_factor = np.max(sobel) / 255
    gradmag = (sobel / scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Calculates the derivatives in x and y directions with a given kernel size
    Then calculates the direction of the resultant vector formed by the x and y gradients
    Uses this direction information and threshold values provided to mask the image
    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    # Apply the following steps to img

    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return binary_output


def gray_select(img, thresh=(0, 255)):
    """

    :param img:
    :param thresh:
    :return:
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_output = np.zeros_like(gray)
    binary_output[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    return binary_output


def rgb_select(img, chn, thresh=(0, 255)):
    """

    :param img:
    :param chn:
    :param thresh:
    :return:
    """

    if chn.upper() == 'B':
        chn_select = img[:, :, 0]
    elif chn.upper() == "G":
        chn_select = img[:, :, 1]
    elif chn.upper() == "R":
        chn_select = img[:, :, 2]
    else:
        sys.exit('Select from (H, L, S) as the channel argument for hls_select() function')

    binary_output = np.zeros_like(chn_select)
    binary_output[(chn_select > thresh[0]) & (chn_select <= thresh[1])] = 1
    return binary_output


def hls_select(img, chn, thresh=(0, 255)):
    """
    Function that converts to HLS color space
    Applies a threshold to the desired channel and returns the mask
    :param img:
    :param chn:
    :param thresh:
    :return:
    """

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    if chn.upper() == 'H':
        chn_select = hls[:, :, 0]
    elif chn.upper() == "L":
        chn_select = hls[:, :, 1]
    elif chn.upper() == "S":
        chn_select = hls[:, :, 2]
    else:
        sys.exit('Select from (H, L, S) as the channel argument for hls_select() function')

    binary_output = np.zeros_like(chn_select)
    binary_output[(chn_select >= thresh[0]) & (chn_select <= thresh[1])] = 1
    return binary_output


def hsv_select(img, chn, thresh=(0, 255)):
    """
    Function that converts to HLS color space
    Applies a threshold to the desired channel and returns the mask
    :param img:
    :param chn:
    :param thresh:
    :return:
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if chn.upper() == 'H':
        chn_select = hsv[:, :, 0]
    elif chn.upper() == "S":
        chn_select = hsv[:, :, 1]
    elif chn.upper() == "V":
        chn_select = hsv[:, :, 2]
    else:
        sys.exit('Select from (H, L, S) as the channel argument for hls_select() function')

    binary_output = np.zeros_like(chn_select)
    binary_output[(chn_select >= thresh[0]) & (chn_select <= thresh[1])] = 1
    return binary_output


def window_mask(width, height, img_ref, center, level):
    """
    Draws boxes
    :param width:
    :param height:
    :param img_ref:
    :param center:
    :param level:
    :return:
    """
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
           max(0, int(center-width/2)):min(int(center+width/2), img_ref.shape[1])] = 1
    return output


def pipeline(img, checkpoint=None):

    # Get the calibration parameters
    calibration_pickle = pickle.load(open('calibration.p', 'rb'))
    mtx = calibration_pickle['mtx']
    dist = calibration_pickle['dist']

    # Undistort the image based on calibration data
    img = cv2.undistort(img, mtx, dist, None, mtx)

    if checkpoint == 'undistorted':
        return img

    # Apply some thresholding methods
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 255))
    # gradmag = mag_thresh(img, sobel_kernel=9, thresh=(30, 100))
    # graddir = dir_threshold(img, sobel_kernel=9, thresh=(0.7, 1.3))

    sthresh = hls_select(img, 's', thresh=(100, 255))
    vthresh = hsv_select(img, 'v', thresh=(50, 255))

    # Get the combined binary
    combined = np.zeros_like(gradx)
    combined[(gradx == 1) & (grady == 1) |
             ((sthresh == 1) & (vthresh == 1))] = 255

    if checkpoint == 'thresholding':
        return combined

    # Apply perspective Transform

    # First video
    # box_width = 0.76
    # mid_width = 0.08
    # height_pct = 0.62
    # bottom_trim = 0.935

    # Second video
    box_width = 0.76
    mid_width = 0.08
    height_pct = 0.62
    bottom_trim = 0.935

    src = np.float32([[img.shape[1] * (.5 - mid_width / 2), img.shape[0] * height_pct],
                      [img.shape[1] * (.5 + mid_width / 2), img.shape[0] * height_pct],
                      [img.shape[1] * (.5 + box_width / 2), img.shape[0] * bottom_trim],
                      [img.shape[1] * (.5 - box_width / 2), img.shape[0] * bottom_trim]])

    # offset adjusts the shrinkage of the warped image - larger is shrunken more
    offset = img.shape[1] * 0.25

    dst = np.float32([[offset, 0],
                      [img.shape[1] - offset, 0],
                      [img.shape[1] - offset, img.shape[0]],
                      [offset, img.shape[0]]])

    # Get perspective transformation matrix and its inverse for later use
    mat = cv2.getPerspectiveTransform(src, dst)
    mat_inv = cv2.getPerspectiveTransform(dst, src)

    if checkpoint == 'straight_warped':

        # Use the matrix for perspective transform
        warped = cv2.warpPerspective(img, mat, (img.shape[1], img.shape[0]),
                                     flags=cv2.INTER_LINEAR)

        return warped, src

    # Use the matrix for perspective transform
    binary_warped = cv2.warpPerspective(combined, mat, (img.shape[1], img.shape[0]),
                                        flags=cv2.INTER_LINEAR)

    if checkpoint == 'warped':

        return binary_warped

    # window settings
    window_width = 25  # consider a window width
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 25  # How much to slide left and right for searching
    smoothing_factor = 15  # Use last 15 results for averaging - smooth the data
    y_scale = 10.0/720.0  # 10 meter is around 720 pixels
    x_scale = 4.0/384.0  # 4 meters is around 384pixels

    # Setup the overall class to do all the tracking

    centorids = LineTracker(window_width=window_width, window_height=window_height,
                            margin=margin, ym=y_scale, xm=x_scale, smooth_factor=smoothing_factor)

    window_centroids = centorids.find_window_centroids(binary_warped)
    # print(window_centroids)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_warped)
        r_points = np.zeros_like(binary_warped)

        # Points used to find the left and right lanes
        rightx = []
        leftx = []

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):

            # add center value found in frame to the list of lane points per left, right
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])

            # window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][1], level)

            # Add graphic points from window mask here to total pixels found

            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        # making the original road pixels 3 color channel
        warpage = np.array(cv2.merge((binary_warped, binary_warped, binary_warped)), np.uint8)
        # overlay the original road image with window results
        output = cv2.addWeighted(warpage, 1.0, template, 0.5, 0.0)

        if checkpoint == 'windows':

            return output

        # fit the lane boundaries to the left, right and center positions found
        yvals = range(0, binary_warped.shape[0])

        # box centers - should be 9 components
        res_yvals = np.arange(binary_warped.shape[0]-(window_height/2), 0, -window_height)

        # fit polynomial to left - 2nd order
        left_fit = np.polyfit(res_yvals, leftx, 2)
        # predict the x value for each y value - continuous with resolution of 1 pixel
        left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
        left_fitx = np.array(left_fitx, np.int32)

        # fit polynomial to right - 2nd order
        right_fit = np.polyfit(res_yvals, rightx, 2)
        # predict the x value for each y value - continuous with resolution of 1 pixel
        right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
        right_fitx = np.array(right_fitx, np.int32)

        # encapsulate lines to give depth
        # left lane
        left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2),
                                                     axis=0),
                                      np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        # right lane
        right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2),
                                                      axis=0),
                                       np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

        # inner lane
        inner_lane = np.array(list(zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] - window_width / 2),
                                                      axis=0),
                                       np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

        # lane lines themselves
        road = np.zeros_like(img)
        cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
        cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
        cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
        # inverse transform to get back to actual perspective
        road_warped = cv2.warpPerspective(road, mat_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        # to get nice outlines
        road_bkg = np.zeros_like(img)
        cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
        cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])
        cv2.fillPoly(road_bkg, [inner_lane], color=[255, 255, 255])
        # inverse transform to get back to actual perspective
        road_warped_bkg = cv2.warpPerspective(road_bkg, mat_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        # return the transformed lane lines
        # first make background black
        base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
        # then add the lane lines
        result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)
        output = result

        # draw the lane lines on an image
        if checkpoint == 'lanelines':

            return output

        # meters per pixel in y direction
        xm_ppx, ym_ppx = centorids.get_ppx_values()
        # print(xm_ppx, ym_ppx)

        # fit a 2nd order polynomial for the actual x and y coordinates of the left lane
        # left lane is more stable
        curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_ppx, np.array(leftx, np.float32) * xm_ppx, 2)
        # using the formula calculate the road curvature
        curverad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_ppx + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * curve_fit_cr[0])
        # print(curverad)

        # calculate the offset of the car on the road
        # average the x pixel values that are closest to the car to find the road center
        road_center = (left_fitx[-1] + right_fitx[-1]) / 2
        # find the difference between the road center and the warped image center - convert it to actual meters
        center_diff = (road_center - binary_warped.shape[1]/2) * xm_ppx
        side_pos = "left"
        if center_diff <= 0:
            # if difference is smaller than zero, warped image center (and hence the car) location
            # is to the right of the road
            side_pos = "right"

        # draw the text showing curvature, offset and speed
        cv2.putText(output, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(output, 'Vehicle is = ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    else:
        output = img

    return output


def pipeline2(img, checkpoint=None):

    # Get the calibration parameters
    calibration_pickle = pickle.load(open('calibration.p', 'rb'))
    mtx = calibration_pickle['mtx']
    dist = calibration_pickle['dist']

    # Undistort the image based on calibration data
    img = cv2.undistort(img, mtx, dist, None, mtx)

    if checkpoint == 'undistorted':
        return img

    # Apply some thresholding methods
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 255))
    # gradmag = mag_thresh(img, sobel_kernel=9, thresh=(30, 100))
    # graddir = dir_threshold(img, sobel_kernel=9, thresh=(0.7, 1.3))

    sthresh = hls_select(img, 's', thresh=(100, 255))
    vthresh = hsv_select(img, 'v', thresh=(50, 255))

    # Get the combined binary
    combined = np.zeros_like(gradx)
    combined[(gradx == 1) & (grady == 1) |
             ((sthresh == 1) & (vthresh == 1))] = 255

    if checkpoint == 'thresholding':
        return combined

    # Apply perspective Transform

    # First video
    # box_width = 0.76
    # mid_width = 0.08
    # height_pct = 0.62
    # bottom_trim = 0.935

    # Second video
    box_width = 0.76
    mid_width = 0.08
    height_pct = 0.62
    bottom_trim = 0.935

    src = np.float32([[img.shape[1] * (.5 - mid_width / 2), img.shape[0] * height_pct],
                      [img.shape[1] * (.5 + mid_width / 2), img.shape[0] * height_pct],
                      [img.shape[1] * (.5 + box_width / 2), img.shape[0] * bottom_trim],
                      [img.shape[1] * (.5 - box_width / 2), img.shape[0] * bottom_trim]])

    # offset adjusts the shrinkage of the warped image - larger is shrunken more
    offset = img.shape[1] * 0.25

    dst = np.float32([[offset, 0],
                      [img.shape[1] - offset, 0],
                      [img.shape[1] - offset, img.shape[0]],
                      [offset, img.shape[0]]])

    # Get perspective transformation matrix and its inverse for later use
    mat = cv2.getPerspectiveTransform(src, dst)
    mat_inv = cv2.getPerspectiveTransform(dst, src)

    if checkpoint == 'straight_warped':

        # Use the matrix for perspective transform
        warped = cv2.warpPerspective(img, mat, (img.shape[1], img.shape[0]),
                                     flags=cv2.INTER_LINEAR)

        return warped, src

    # Use the matrix for perspective transform
    binary_warped = cv2.warpPerspective(combined, mat, (img.shape[1], img.shape[0]),
                                        flags=cv2.INTER_LINEAR)

    if checkpoint == 'warped':

        return binary_warped


    # apply

    return output


def test_pipeline(test_folder, out_folder, checkpoint=None):

    file_name_pattern = test_folder + '/test*.jpg'
    images = glob.glob(file_name_pattern)

    for idx, img_name in enumerate(images):

        print("Working on test image # ", idx + 1)

        img = cv2.imread(img_name)

        result = pipeline(img, checkpoint)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=15)
        if len(result.shape) == 3:
            ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            ax2.imshow(result, cmap="gray")
        ax2.set_title('Processed Image', fontsize=15)
        plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.)
        img_name = os.path.split(os.path.splitext(img_name)[0])[1]
        plt.savefig(os.path.join(out_folder, str(img_name + '_compare')))
        plt.close()


def test_perspective_transform(test_folder, out_folder, checkpoint='straight_warped'):

    file_name_pattern = test_folder + '/straight_lines*.jpg'
    images = glob.glob(file_name_pattern)

    for idx, img_name in enumerate(images):

        print("Working on straight test image # ", idx + 1)

        img = cv2.imread(img_name)
        img_name = os.path.split(os.path.splitext(img_name)[0])[1]

        if checkpoint == 'straight_warped':
            result, src = pipeline(img, checkpoint)

            polygon = patches.Polygon(src, closed=True, alpha=0.5, color='g')

            fig, ax = plt.subplots(1)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.add_patch(polygon)
            plt.savefig(os.path.join(out_folder, str(img_name + '_points')))
            plt.close()
        else:
            result = pipeline(img, checkpoint)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=15)
        ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax2.set_title('Processed Image', fontsize=15)
        plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.)
        plt.savefig(os.path.join(out_folder, str(img_name + '_compare')))
        plt.close()


def process_video(file_path, out_folder):

    input_video = file_path
    output_video = os.path.split(os.path.splitext(file_path)[0])[1]
    output_video = os.path.join(out_folder, str(output_video + '_processed.mp4'))

    clip1 = VideoFileClip(input_video)
    video_clip = clip1.fl_image(pipeline)
    video_clip.write_videofile(output_video, audio=False)