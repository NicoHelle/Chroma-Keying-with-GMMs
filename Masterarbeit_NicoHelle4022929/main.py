
import cv2
import numpy as np
import PyQt5
from PyQt5.QtWidgets import QApplication
import sys
import os
import warnings
import time
from gui import ChromaKeyingApplication
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plot
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
from preprocess import *
from ChromaKeying import *
import traceback
from postprocess import *
from plotandvisualize import *
from gui import ChromaKeyingApplication
from preprocess import *
from ChromaKeying import *


os.environ["OMP_NUM_THREADS"] = '4'
#Used to ignore the Kmeans warning when computing the GMM
warnings.filterwarnings('ignore')


#Used to track reported error message to not get duplicate error messages while computing the main pipeline
error_reported = False

#Variable used to track if the mask is updated
update_mask = True

#Used to track printed console output
current_console_output = set()
last_console_output = None

#Initialises the webcam, comment in/out the preferred initialisation Method

#Uses DSHOW as initialisation method -> faster but less stability
webcam_frame = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Uses MSMF as initialisation method -> slower but more stability
#webcam_frame  = cv2.VideoCapture(0, cv2.CAP_MSMF)

#Change the frame size, 640x480 is the default value
#webcam_frame.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#webcam_frame.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Main function gets exectued when starting the programm
def main():
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()

        if cuda_count > 0:
            print(f"CUDA is available for computation")
        else: print(f"No CUDA module found, functions which require CUDA will not work!!!")

        #Starts the Application
        app = QApplication(sys.argv)
        window = ChromaKeyingApplication()
        print_once('Started application...')

        #Connects the corresponding Functions with the GUI
        window.plot_alphablending_samples.clicked.connect(lambda: plot_Standard_ChromaKeying(window))
        window.update_GMM_or_BGMM.clicked.connect(lambda: selected_GMM_Method(window))

        #Updates the frame continously, realizes the program pipeline: Brightness/Contrastadjustment, Preprocessing
        #ChromaKeying, Postprocessing, frame display
        def process_pipeline():
            #Captues the frame and returns nothing when the frame could not be read successfully
            frame_read, frame = webcam_frame.read()
            if not frame_read:
                return

            #Starts the time measurment for displaying processing times of pipeline elements in the GUI
            function_times = {}
            function_start_time = time.time()

            #Converts the Frame to the YUV Color Space
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            #Optional in GUI: Apply Preprocess Filter (Noise Filter)
            processed_frame = filter_noise(window, frame)
            function_times["Preprocessing"] = time.time() - function_start_time
            print_once('Applied preprocessing...')

            #Optional in GUI: Apply brightness and contrast adjustments
            processed_frame = frame_brightness_contrast_adjustment(processed_frame, window.brightness_slider.value(), window.contrast_slider.value())
            print_once('Applied Brightness/Contrast adjustments...')

            #Copies the processed frame to make it usable in the GUI and other modules
            window.current_frame = processed_frame

            #Applies the specified Chroma Keying method.
            function_start_time = time.time()
            # Applies GMM Chromakeying if GMM_ChromaKey_Checkbox is checked
            if window.GMM_ChromaKey_Checkbox.isChecked():
                print_once('Using GMM chroma keying...')
                frame = GMM_ChromaKeying(window, processed_frame, frame, new_samples=False)
                processed_frame = overlay_sample_areas(processed_frame, window)
            # Applies BGMM Chromakeying if BGMM_ChromaKey_Checkbox is checked
            elif window.BGMM_ChromaKey_Checkbox.isChecked():
                print_once('Using Bayesian GMM chroma keying...')
                frame = BGMM_ChromaKeying(window, processed_frame, frame, new_samples=False)
                processed_frame = overlay_sample_areas(processed_frame, window)
            # Applies Standard Chromakeying if Standard_ChromaKey_Checkbox is checked
            elif window.Standard_ChromaKey_Checkbox.isChecked():
                print_once('Using Standard chroma keying...')
                frame = Standard_ChromaKeying(window, processed_frame, frame)

            #processing times of the Chroma Keying functions
            function_times["Chroma Keying"] = time.time() - function_start_time
            print_once("Chroma keyed the image...")

            # Display original frame
            function_start_time = time.time()
            window.display_original_frame(window.original_label, frame)
            print_once("Original frame displayed...'")

            # Display processed frame
            window.display_preprocessed_frame(window.processed_label, processed_frame)
            print_once('Processed frame displayed...')
            function_times["Display Update"] = time.time() - function_start_time

            #Updates the GUI processing times
            window.update_timing_label(function_times)
    #Error Handling
    except Exception as e:
        global error_reported
        if not error_reported:
            print("An Error occured in Main or process_pipeline:", e)
            print(traceback.format_exc())
            error_reported = True

    #connects the GUI process_pipeline method to the Main process_pipeline method
    window.process_pipeline = process_pipeline

    #displays the window
    window.show()

    sys.exit(app.exec_())

#Takes Samples from background, forgeground and inbetween Areas of the ChromaKeyed image to visualize the result
def plot_Standard_ChromaKeying(window):
    try:
        #Copies the original_frame and processed_frame and uses them to compute the mask
        original_frame = getattr(window, 'current_frame', None)
        processed_frame = window.current_frame.copy()
        mask = return_mask_chromakey(window, processed_frame, original_frame)

        #Number of Samples per Category (Foreground, inbetween, Background)
        sample_size = 500

        #Gets all Indices of Pixels where the Constraints apply:
        # mask = 1  -> Foreground, mask = 0 -> Background, Values between 0 and 1 -> unsure (inbetween)
        foreground = np.argwhere(mask == 1)
        inbetween = np.argwhere((mask > 0) & (mask < 1))
        background = np.argwhere(mask == 0)

        #Only print the Samples when there is at least 1 Sample for every category
        if len(inbetween) == 0 or len(background) == 0 or len(foreground) == 0:
            print("At least 1 Sample point is necessary for every Category of Classification (Foreground, Background, inbetween")
            return

        #Take up to sample_size Samples from each category
        foreground_samples = foreground[
            np.random.choice(len(foreground), min(sample_size, len(foreground)))
        ]
        background_samples = background[
            np.random.choice(len(background), min(sample_size, len(background)))
        ]
        inbetween_samples = inbetween[
            np.random.choice(len(inbetween), min(sample_size, len(inbetween)))
        ]

        #Extract the YUV-Values of the sample-indices from the frame
        YUV_foreground = original_frame[foreground_samples[:, 0], foreground_samples[:, 1]]
        YUV_background = original_frame[background_samples[:, 0], background_samples[:, 1]]
        YUV_inbetween = original_frame[inbetween_samples[:, 0], inbetween_samples[:, 1]]

        #Use Scatterplot to visualize the samples
        figure = plot.figure()
        axis = figure.add_subplot(111, projection='3d')
        axis.scatter(YUV_foreground[:, 0], YUV_foreground[:, 1], YUV_foreground[:, 2], label='Foreground', c='red', marker='o')
        axis.scatter(YUV_background[:, 0], YUV_background[:, 1], YUV_background[:, 2], label='Background', c='green', marker='x')
        axis.scatter(YUV_inbetween[:, 0], YUV_inbetween[:, 1], YUV_inbetween[:, 2], label='Inbetween', c='blue', marker='^')

        #Y,U,V Values of the Sliders
        y_minimum = window.y_min_slider.value()
        y_maximum = window.y_max_slider.value()
        u_minimum = window.u_min_slider.value()
        u_maximum = window.u_max_slider.value()
        v_minimum = window.v_min_slider.value()
        v_maximum = window.v_max_slider.value()

        #Defines the corner points of the cupoid
        cuboid_corner_points = np.array([[y_minimum, u_minimum, v_minimum],[y_maximum, u_minimum, v_minimum],
                                         [y_maximum, u_maximum, v_minimum], [y_minimum, u_maximum, v_minimum],
                                         [y_minimum, u_minimum, v_maximum], [y_maximum, u_minimum, v_maximum],
                                         [y_maximum, u_maximum, v_maximum], [y_minimum, u_maximum, v_maximum]
                                         ])

        #Defines the edges between the cupoid_corner points
        cuboid_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),  (0, 4), (1, 5), (2, 6), (3, 7)]

        #Drawing the edges of the cupoid
        for i, cuboid_edge in enumerate(cuboid_edges):
            cuboid_points = cuboid_corner_points[list(cuboid_edge)]
            axis.plot(cuboid_points[:, 0], cuboid_points[:, 1], cuboid_points[:, 2], label='Classification Boundaries' if i == 0 else "", color='black', linewidth=3)

        #Label the axis
        axis.set_xlabel('Y')
        axis.set_ylabel('U')
        axis.set_zlabel('V')
        axis.legend()
        plot.show()
        print("Plotted the Standard_ChromaKeying Values")
    except Exception as e:
        print("An Error occured in plot_Standard_ChromaKeying:", e)
        print(traceback.format_exc())

#Used to print out the pipeline without repating old console output.
def print_once(console_output):
    global last_console_output
    if console_output not in current_console_output:
        if console_output != last_console_output:
            print(console_output)
            current_console_output.add(console_output)
            last_console_output = console_output

#Used to compute the GMM in regard to whether GMM or BGMM is selceted in the GUI when pressing the Update GMM/BGMM Values Button
def selected_GMM_Method(window):
    if window.BGMM_ChromaKey_Checkbox.isChecked():
        BGMM_ChromaKeying(window, None, None, new_samples=True)
    else:
        GMM_ChromaKeying(window, None, None, new_samples=True)

if __name__ == '__main__':
    main()
    webcam_frame.release()
