import cv2
import numpy as np
from collections import deque
import traceback
import os

#Overlays the drawn sample shapes on the processed_image for visualization
def overlay_sample_areas(processed_frame, window):
    try:
        #When window has no attribute drawn_sample_areas a new Queue with that name is generated
        #Queue length = maximum_components_slider.value() * 2 because of a bug that occurs when
        #not multiplying by 2.
        if not hasattr(window, 'drawn_sample_areas'):
            window.drawn_sample_areas = deque(maxlen=window.maximum_components_slider.value() * 2)

        #Update the Queue Size (so that length = number components)
        max_queue_size = window.maximum_components_slider.value()
        window.drawn_sample_areas = deque(window.drawn_sample_areas, maxlen=max_queue_size * 2)

        #Create copy of processed frame
        overlay = processed_frame.copy()

        #removes all duplicate shapes
        unique_sample_areas = set(map(tuple, window.drawn_sample_areas))

        #draw the shapes onto the overlay
        for sample_area in unique_sample_areas:
            cv2.polylines(overlay, [np.array(sample_area)], isClosed=True, color=(0, 255, 0), thickness=2)

        #Blend the overlay and the processed_frame together
        result = cv2.addWeighted(processed_frame, 0.5, overlay, 0.5, 0)

        return result
    except Exception as e:
        print("An error occured in overlay_sample_areas:", e)
        print(traceback.format_exc())
        return processed_frame

#Used to compute the alpha matte with the binary mask computed by the GMM/BGMM
def compute_alpha_matte(original_frame, mask, window=None):

    #Custom background image
    background = 'test.jpg'

    try:
        # Optional in GUI: Apply Erosion to mask
        if window.use_erosion.isChecked():
            erosion_mask = np.ones((window.erosion_slider.value(), window.erosion_slider.value()), np.uint8)
            mask = cv2.erode(mask, erosion_mask)

        # Optional in GUI: Apply dilation to mask
        if window.use_dilation.isChecked():
            dilation_mask = np.ones((window.dilation_slider.value(), window.dilation_slider.value()), np.uint8)
            mask = cv2.dilate(mask, dilation_mask)

        #Detect edges with Canny filter
        edge_image = cv2.Canny(mask, 100, 150)
        #cv2.imshow("Edges", edges)

        #Blur edges for smoother transitions between foreground and background
        blur_edge_image = cv2.GaussianBlur(edge_image.astype(np.float32), (5, 5), 0) / 255.0
        #cv2.imshow("Blurred Edges", blurred_edges)

        # Compute Adaptive Alpha Matte Based on Edges
        alpha_matte = np.clip(mask / 255.0 + blur_edge_image, 0, 1)
        #cv2.imshow("Alpha blending", alpha_matte)

        # Load and resize the background image to match frame size
        background = os.path.join(os.path.dirname(__file__), 'test.jpg')
        background = cv2.cvtColor(cv2.imread(background), cv2.COLOR_BGR2YUV)
        background = cv2.resize(background, (original_frame.shape[1], original_frame.shape[0]))

        # Blend the Foreground and the background using the computed alpha matte
        foreground = original_frame.astype(np.float32)
        background = background.astype(np.float32)
        result = (foreground * alpha_matte[..., None] + background * (1 - alpha_matte[..., None])).astype(np.uint8)

        #Used for testing the alpha matte
        #result[uncertain_mask] = [255, 0, 255]

        return result
    except Exception as e:
        print("An error occured in compute_alpha_matte:", e)
        print(traceback.format_exc())
        return original_frame

