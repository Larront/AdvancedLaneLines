import numpy as np
import cv2


class LineTracker:

    def __init__(self, window_width, window_height, margin, ym=1, xm=1, smooth_factor=15):

        # list that stores all the past (left, right) center set values used for smoothing the output
        self.__recent_centers = []

        # the window pixel width of the center values, used to count pixels inside center windows
        # to determine curve values
        self.__window_width = window_width

        # the window pixel height of the center values, used to count pixels inside center windows
        # to determine curve values. breaks the image into vertical levels
        self.__window_height = window_height

        # the pixel distance in both directions to slide (left_window + right_window) template for searching
        # padding
        self.__margin = margin

        # converting for real world values
        # meters per pixel in vertical axis
        self.__ymppx = ym

        # meters per pixel in horizontal axis
        self.__xmppx = xm

        # smoothing factor
        # past 15 values
        self.__smooth_factor = smooth_factor

    def get_ppx_values(self):
        return self.__xmppx, self.__ymppx

    # the main tracking function for finding and storing lane segment positions
    def find_window_centroids(self, warped):

        width = self.__window_width
        height = self.__window_height
        margin = self.__margin

        # store the (left, right) window centroid positions per level
        window_centroids = []
        window = np.ones(width)  # Create our window template that we will use for convolutions

        # first find the two starting positions for the left and right lane by using to get the vertical image slide
        # and then np.convolve the vertical image slide with the window template

        # sum the bottom quarter of the image
        # first do it for the left side of the bottom quarter, sum for each column
        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        # find the maximum value after convolution with the window which is all ones
        # the peak will be a plateau with a length of window width and argmax will find the rightmost point
        # subtract half of the window width to get the center;
        l_center = np.argmax(np.convolve(window, l_sum)) - width/2
        # do the summation for the right side of the bottom quarter of the image
        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        # now after subtracting half of the window width add the first half of the position
        r_center = np.argmax(np.convolve(window, r_sum)) - width / 2 + int(warped.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(warped.shape[0]/height)):

            # convolve the image into the vertical slice of the image
            img_layer = np.sum(warped[int(warped.shape[0] - (level + 1) * height):int(warped.shape[0] - level * height), :],
                                 axis=0)
            conv_signal = np.convolve(window, img_layer)

            # find the best left centroid by using past left center as a reference
            # use width/2 as offset because convolution signal reference is at right side of window, not center ofwindow
            # look only to the left and right side of the previous maximum

            offset = width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        self.__recent_centers.append(window_centroids)

        # return average values of the line centers, helps to keep the markers from jumping around too much
        # let's look at past 15 values (default for smooth factor)
        return np.average(self.__recent_centers[-self.__smooth_factor:], axis=0)
