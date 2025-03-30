import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QSlider, QButtonGroup, QGroupBox, QTextEdit, QPushButton,QRadioButton,QScrollArea,QSizePolicy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2
from collections import deque
import traceback

error_reported = False

#Uses QWidget to create the GUI
class ChromaKeyingApplication(QWidget):
    #initialize the GUI
    def __init__(self):
        super().__init__()

        #Builds the user Interface
        self.SetupGUI()
        # Number of sample Sets
        self.sample_sets = deque(maxlen=1)
        # counts how many Sample Sets that were drawn
        self.counter = 0

    #Defines all the sliders, labels, checkboxes, radio buttons, timing results etc.
    def SetupGUI(self):
        try:
            #Title of the Application Window
            self.setWindowTitle("Application for GMM/BGMM ChromaKeying")

            #The MainLayout is adding GUI Elements vertically (from top to the bottom)
            MainLayout = QVBoxLayout()
            # The VideoLayout is adding GUI Elements horizontally (from left to right)
            VideoLayout = QHBoxLayout()

            #--------------------------------------------------------------------------------------------
            #Brightness Labels/Sliders

            #region Brightness and Contrast
            #Brightness and Contrast Layout
            brightness_and_contrast_layout = QHBoxLayout()

            #Brightness Slider and Label
            self.brightness_label = QLabel("Brightness: 0")
            self.brightness_slider = QSlider(Qt.Horizontal)
            self.brightness_slider.setRange(-75, 75)
            self.brightness_slider.setValue(0)

            #Connects the sliders/label to the update_label function
            self.brightness_slider.valueChanged.connect(
                lambda: self.update_label(self.brightness_slider, self.brightness_label, "Brightness: "))

            #Add to Layout
            brightness_and_contrast_layout.addWidget(self.brightness_label)
            brightness_and_contrast_layout.addWidget(self.brightness_slider)

            # --------------------------------------------------------------------------------------------
            #Contrast Labels/Sliders

            self.contrast_label = QLabel("Contrast: 0")
            self.contrast_slider = QSlider(Qt.Horizontal)
            self.contrast_slider.setRange(-75, 75)
            self.contrast_slider.setValue(0)

            #Connects the sliders/label to the update_label function
            self.contrast_slider.valueChanged.connect(
                lambda: self.update_label(self.contrast_slider, self.contrast_label, "Contrast: "))

            #Add to Layout
            brightness_and_contrast_layout.addWidget(self.contrast_label)
            brightness_and_contrast_layout.addWidget(self.contrast_slider)

            #Add Brightness and Contrast Layout to Main Layout
            MainLayout.addLayout(brightness_and_contrast_layout)

            #endregion

            # --------------------------------------------------------------------------------------------
            # Bilateral filter labels/sliders
            #region Bilateral Filter
            bilateral_layout = QHBoxLayout()

            # Checkbox for applying the Bilateral filter
            self.bilateral_filter_checkbox = QCheckBox("Use Bilateral Filter")
            self.bilateral_filter_checkbox.stateChanged.connect(self.make_bilateral_sliders_visible)
            bilateral_layout.addWidget(self.bilateral_filter_checkbox)

            #Fixed width for labels and sliders to fit them into the window
            label_width = 100
            slider_width = 150

            #Bilateral Kernel Sliders/Label
            self.bilateral_kernel_size_label = QLabel("Kernel: 10")
            self.bilateral_kernel_size_label.setFixedWidth(label_width)
            self.bilateral_kernel_size = QSlider(Qt.Horizontal)
            self.bilateral_kernel_size.setRange(3, 20)
            self.bilateral_kernel_size.setValue(10)
            self.bilateral_kernel_size.setFixedWidth(slider_width)

            #Connect to the update_label function
            self.bilateral_kernel_size.valueChanged.connect(
                lambda: self.update_label(self.bilateral_kernel_size, self.bilateral_kernel_size_label, "Kernel: "))

            # Bilateral Sigma Color Sliders/Labels
            self.bilateral_sigma_color_label = QLabel("Sigma Color: 50")
            self.bilateral_sigma_color_label.setFixedWidth(label_width)
            self.bilateral_sigma_color = QSlider(Qt.Horizontal)
            self.bilateral_sigma_color.setRange(1, 100)
            self.bilateral_sigma_color.setValue(50)
            self.bilateral_sigma_color.setFixedWidth(slider_width)
            self.bilateral_sigma_color.valueChanged.connect(
                lambda: self.update_label(self.bilateral_sigma_color, self.bilateral_sigma_color_label,
                                                "Sigma Color: "))

            #Bilateral Sigma Space Sliders/Labels
            self.bilateral_sigma_space_label = QLabel("Sigma Space: 50")
            self.bilateral_sigma_space_label.setFixedWidth(label_width)
            self.bilateral_sigma_space = QSlider(Qt.Horizontal)
            self.bilateral_sigma_space.setRange(1, 100)
            self.bilateral_sigma_space.setValue(50)
            self.bilateral_sigma_space.setFixedWidth(slider_width)
            self.bilateral_sigma_space.valueChanged.connect(
                lambda: self.update_label(self.bilateral_sigma_space, self.bilateral_sigma_space_label,
                                                "Sigma Space: "))

            #Hide the sliders/labels when the bilateral_filter_checkbox is not checked yet
            bilateral_layout.addWidget(self.bilateral_kernel_size_label)
            bilateral_layout.addWidget(self.bilateral_kernel_size)
            bilateral_layout.addWidget(self.bilateral_sigma_color_label)
            bilateral_layout.addWidget(self.bilateral_sigma_color)
            bilateral_layout.addWidget(self.bilateral_sigma_space_label)
            bilateral_layout.addWidget(self.bilateral_sigma_space)
            self.bilateral_kernel_size_label.hide()
            self.bilateral_kernel_size.hide()
            self.bilateral_sigma_color_label.hide()
            self.bilateral_sigma_color.hide()
            self.bilateral_sigma_space_label.hide()
            self.bilateral_sigma_space.hide()

            # Add the bilateral layout to the main layout
            MainLayout.addLayout(bilateral_layout)
            # endregion

            #CUDA bilateral filter sliders/label
            # --------------------------------------------------------------------------------------------
            #region CUDA Bilateral Filter

            cuda_bilateral_layout = QHBoxLayout()

            # Checkbox for applying the CUDA Bilateral filter
            self.cuda_bilateral_filter_checkbox = QCheckBox("Use CUDA Bilateral Filter")
            self.cuda_bilateral_filter_checkbox.stateChanged.connect(self.make_cuda_bilateral_filter_sliders_visible)
            cuda_bilateral_layout.addWidget(self.cuda_bilateral_filter_checkbox)

            # CUDA Bilateral kernel sliders/labels
            self.cuda_bilateral_kernel_size_label = QLabel("Kernel: 15")
            self.cuda_bilateral_kernel_size_label.setFixedWidth(label_width)
            self.cuda_bilateral_kernel_size = QSlider(Qt.Horizontal)
            self.cuda_bilateral_kernel_size.setRange(5, 25)
            self.cuda_bilateral_kernel_size.setValue(15)
            self.cuda_bilateral_kernel_size.setFixedWidth(slider_width)
            self.cuda_bilateral_kernel_size.valueChanged.connect(
                lambda: self.update_label(self.cuda_bilateral_kernel_size, self.cuda_bilateral_kernel_size_label, "Kernel: "))

            # CUDA Bilateral sigma color sliders/labels
            self.cuda_bilateral_color_label = QLabel("Sigma Color: 50")
            self.cuda_bilateral_color_label.setFixedWidth(label_width)
            self.cuda_bilateral_color = QSlider(Qt.Horizontal)
            self.cuda_bilateral_color.setRange(1, 150)
            self.cuda_bilateral_color.setValue(75)
            self.cuda_bilateral_color.setFixedWidth(slider_width)
            self.cuda_bilateral_color.valueChanged.connect(
                lambda: self.update_label(self.cuda_bilateral_color, self.cuda_bilateral_color_label,
                                                "Sigma Color: "))

            # CUDA Bilateral sigma space sliders/labels
            self.cuda_bilateral_space_label = QLabel("Sigma Space: 75")
            self.cuda_bilateral_space_label.setFixedWidth(label_width)
            self.cuda_bilateral_space = QSlider(Qt.Horizontal)
            self.cuda_bilateral_space.setRange(1, 150)
            self.cuda_bilateral_space.setValue(75)
            self.cuda_bilateral_space.setFixedWidth(slider_width)
            self.cuda_bilateral_space.valueChanged.connect(
                lambda: self.update_label(self.cuda_bilateral_space, self.cuda_bilateral_space_label,
                                                "Sigma space: "))

            #Hide the sliders/labels when the cuda_bilateral_filter_checkbox is not checked yet
            cuda_bilateral_layout.addWidget(self.cuda_bilateral_kernel_size_label)
            cuda_bilateral_layout.addWidget(self.cuda_bilateral_kernel_size)
            cuda_bilateral_layout.addWidget(self.cuda_bilateral_color_label)
            cuda_bilateral_layout.addWidget(self.cuda_bilateral_color)
            cuda_bilateral_layout.addWidget(self.cuda_bilateral_space_label)
            cuda_bilateral_layout.addWidget(self.cuda_bilateral_space)
            self.cuda_bilateral_kernel_size_label.hide()
            self.cuda_bilateral_kernel_size.hide()
            self.cuda_bilateral_color_label.hide()
            self.cuda_bilateral_color.hide()
            self.cuda_bilateral_space_label.hide()
            self.cuda_bilateral_space.hide()

            #Add the CUDA Bilateral layout to the main layout
            MainLayout.addLayout(cuda_bilateral_layout)
            #endregion

            #NL-means sliders/label
            #-----------------------------------------------------------------------------------------------------
            #region NLCUDA Filter Layout (QHBoxLayout for inline arrangement)
            NLmeans_layout = QHBoxLayout()

            # Checkbox for applying the NLCUDA filter
            self.NLmeans_checkbox = QCheckBox("Use Non-Local Means Filter")
            self.NLmeans_checkbox.stateChanged.connect(self.make_nlmneans_slider_visible)
            NLmeans_layout.addWidget(self.NLmeans_checkbox)

            # Label/Sliders for color filter strength
            self.color_filter_label = QLabel("Filter Luminance: 5")
            self.color_filter_label.setFixedWidth(label_width)
            self.color_filter_slider = QSlider(Qt.Horizontal)
            self.color_filter_slider.setRange(1, 50)
            self.color_filter_slider.setValue(5)
            self.color_filter_slider.setFixedWidth(slider_width)
            self.color_filter_slider.valueChanged.connect(
                lambda: self.update_label(self.color_filter_slider, self.color_filter_label, "Filter Luminance: "))

            # Label/Sliders for luminance filter strength
            self.luminance_filter_label = QLabel("Filter Color: 5")
            self.luminance_filter_label.setFixedWidth(label_width)
            self.luminance_filter_slider = QSlider(Qt.Horizontal)
            self.luminance_filter_slider.setRange(1, 50)
            self.luminance_filter_slider.setValue(5)
            self.luminance_filter_slider.setFixedWidth(slider_width)
            self.luminance_filter_slider.valueChanged.connect(
                lambda: self.update_label(self.luminance_filter_slider, self.luminance_filter_label, "Filter Color: "))

            #Hide the sliders/labels when the NLmeans_checkbox is not checked yet
            NLmeans_layout.addWidget(self.color_filter_label)
            NLmeans_layout.addWidget(self.color_filter_slider)
            NLmeans_layout.addWidget(self.luminance_filter_label)
            NLmeans_layout.addWidget(self.luminance_filter_slider)
            self.color_filter_label.hide()
            self.color_filter_slider.hide()
            self.luminance_filter_label.hide()
            self.luminance_filter_slider.hide()

            # Add the NLmeans layout to the main layout
            MainLayout.addLayout(NLmeans_layout)
            #endregion

            # -----------------------------------------------------------------------------------------------------
            #region Multi-Frame Denoising Controls
            # Multi-Frame Denoising Section
            multi_denoise_layout = QHBoxLayout()

            # Checkbox for applying multi-frame denoising
            self.multiFrameDenoise_Checkbox = QCheckBox("Use Multi-Frame Denoising")
            self.multiFrameDenoise_Checkbox.stateChanged.connect(self.make_multi_denoise_sliders_visible)
            multi_denoise_layout.addWidget(self.multiFrameDenoise_Checkbox)

            # Slider/Labels for Luminance Filter Strength
            self.multi_luminance_label = QLabel("Luminance Filter: 25")
            self.multi_luminance_slider = QSlider(Qt.Horizontal)
            self.multi_luminance_slider.setRange(0,100)
            self.multi_luminance_slider.setValue(25)
            self.multi_luminance_slider.valueChanged.connect(
                lambda: self.update_label(self.multi_luminance_slider, self.multi_luminance_label,
                                          "Luminance Filter: "))

            multi_denoise_layout.addWidget(self.multi_luminance_label)
            multi_denoise_layout.addWidget(self.multi_luminance_slider)

            #Hide the sliders/labels when the multiFrameDenoise_Checkbox is not checked yet
            self.multi_luminance_label.hide()
            self.multi_luminance_slider.hide()

            # Slider/Labels for Color Filter Strengh
            self.multi_color_label = QLabel("Color Filter: 25")
            self.multi_color_slider = QSlider(Qt.Horizontal)
            self.multi_color_slider.setRange(0, 100)
            self.multi_color_slider.setValue(25)
            self.multi_color_slider.valueChanged.connect(
                lambda: self.update_label(self.multi_color_slider, self.multi_color_label,
                                          "Color Filter: "))

            #Hide the sliders/labels when the multiFrameDenoise_Checkbox is not checked yet
            self.multi_color_label.hide()
            self.multi_color_slider.hide()

            #Add Elements to the Layout
            multi_denoise_layout.addWidget(self.multi_color_label)
            multi_denoise_layout.addWidget(self.multi_color_slider)

            # Add the multi-frame denoising layout to the main layout
            MainLayout.addLayout(multi_denoise_layout)
            #endregion

            # -----------------------------------------------------------------------------------------------------
            # create radio button group for the chroma key options
            self.ChromaKeyingGroup = QButtonGroup(self)
            self.ChromaKeyingGroup.setExclusive(True)

            # Create radio buttons for GMM_ChromaKeying, BGMM_Chromakeying, and Standard_ChromaKeying.
            self.GMM_ChromaKey_Checkbox = QRadioButton("Use Standard GMM")
            self.BGMM_ChromaKey_Checkbox = QRadioButton("Use Bayesian GMM")
            self.Standard_ChromaKey_Checkbox = QRadioButton("Use Standard Chroma Keying")

            #default selection
            self.GMM_ChromaKey_Checkbox.setChecked(True)

            #Add the radio buttons to the group and the MainLayout
            self.ChromaKeyingGroup.addButton(self.GMM_ChromaKey_Checkbox)
            self.ChromaKeyingGroup.addButton(self.BGMM_ChromaKey_Checkbox)
            self.ChromaKeyingGroup.addButton(self.Standard_ChromaKey_Checkbox)
            MainLayout.addWidget(self.GMM_ChromaKey_Checkbox)
            MainLayout.addWidget(self.BGMM_ChromaKey_Checkbox)
            MainLayout.addWidget(self.Standard_ChromaKey_Checkbox)

            # -----------------------------------------------------------------------------------------------------
            # regionMaximum Queue Length
            # Sliders/Labels for maximum_components
            self.maximum_components_label = QLabel("Maximum Components")
            self.maximum_components_slider = QSlider(Qt.Horizontal)
            self.maximum_components_slider.setMinimum(1)
            self.maximum_components_slider.setMaximum(15)
            self.maximum_components_slider.setValue(1)
            self.maximum_components_slider.valueChanged.connect(self.update_maximum_queque_length)

            # Add Sliders/Labels to the Layout
            MainLayout.addWidget(self.maximum_components_label)
            MainLayout.addWidget(self.maximum_components_slider)

            #Generates sample_set Queue
            self.sample_sets = deque(maxlen=self.maximum_components_slider.value())
            self.counter = 0
            # endregion

            # -----------------------------------------------------------------------------------------------------
            # Slider/Labels for log-likelihood Threshold
            self.threshold_label = QLabel("Threshold")
            self.threshold_slider = QSlider(Qt.Horizontal)
            self.threshold_slider.setMinimum(-1000)
            self.threshold_slider.setMaximum(0)
            self.threshold_slider.setValue(-20)
            self.threshold_slider.valueChanged.connect(self.update_threshold_label)

            # Add elements to MainLayout
            MainLayout.addWidget(self.threshold_label)
            MainLayout.addWidget(self.threshold_slider)

            # -----------------------------------------------------------------------------------------------------
            # Chroma Keying Sliders Group Box
            self.chroma_sliders_group = QGroupBox("Chroma Keying Range Sliders")
            self.chroma_sliders_layout = QVBoxLayout()
            self.chroma_sliders_group.setLayout(self.chroma_sliders_layout)
            MainLayout.addWidget(self.chroma_sliders_group)

            self.y_min_slider = self.create_slider("Y Min: ", 0, 255)
            self.y_max_slider = self.create_slider("Y Max: ", 0, 255)
            self.u_min_slider = self.create_slider("U Min: ", 0, 255)
            self.u_max_slider = self.create_slider("U Max: ", 0, 255)
            self.v_min_slider = self.create_slider("V Min: ", 0, 255)
            self.v_max_slider = self.create_slider("V Max: ", 0, 255)

            #endregion

            # -----------------------------------------------------------------------------------------------------
            #Labels and Sliders for Erosion/Dilation
            #region Erosion and Dilation Controls
            erosion_dilation_layout = QHBoxLayout()

            #Checkbox for the user to enable/disable erosion
            self.use_erosion = QCheckBox("Use Erosion")
            erosion_dilation_layout.addWidget(self.use_erosion)

            #Erosion Kernel Slider/Labels
            self.erosion_label = QLabel("Kernel Size: 1")
            self.erosion_slider = QSlider(Qt.Horizontal)
            self.erosion_slider.setRange(1, 8)
            self.erosion_slider.setValue(1)
            self.erosion_slider.valueChanged.connect(
                lambda: self.update_label(self.erosion_slider, self.erosion_label,
                                                "Kernel Size: "))

            #Add Layout to the MainLayout
            erosion_dilation_layout.addWidget(self.erosion_label)
            erosion_dilation_layout.addWidget(self.erosion_slider)

            # Checkbox for the user to enable/disable dilation
            self.use_dilation = QCheckBox("Use Dilation")
            erosion_dilation_layout.addWidget(self.use_dilation)

            #Dilation Kernel Slider/Labels
            self.dilation_label = QLabel("Kernel Size: 1")
            self.dilation_slider = QSlider(Qt.Horizontal)
            self.dilation_slider.setRange(1, 8)
            self.dilation_slider.setValue(1)
            self.dilation_slider.valueChanged.connect(
                lambda: self.update_label(self.dilation_slider, self.dilation_label,
                                                "Kernel Size: "))

            # Add Layout to the MainLayout
            erosion_dilation_layout.addWidget(self.dilation_label)
            erosion_dilation_layout.addWidget(self.dilation_slider)

            #Add erosion and dilation layout to MainLayout
            MainLayout.addLayout(erosion_dilation_layout)
            #endregion

            # -----------------------------------------------------------------------------------------------------
            #Button for plotting the YUV_Values of Standard_ChromaKeying
            #region PlotSamples
            self.plot_alphablending_samples = QPushButton("Plot Chroma Keying Values")
            MainLayout.addWidget(self.plot_alphablending_samples)
            #endregion

            # Button for updating the GMM/BGMM with new samples
            self.update_GMM_or_BGMM = QPushButton("Update GMM/BGMM Samples")
            MainLayout.addWidget(self.update_GMM_or_BGMM)

            # -----------------------------------------------------------------------------------------------------
            #region Video Display Labels
            self.original_label = QLabel("Chroma Keyed Video")
            self.processed_label = QLabel("Processed Video")
            VideoLayout.addWidget(self.original_label)
            VideoLayout.addWidget(self.processed_label)
            #endregion

            # -----------------------------------------------------------------------------------------------------
            #region FPS and Processing Time Labels
            self.timing_label = QLabel("Processing Times:")
            MainLayout.addWidget(self.timing_label)

            # -----------------------------------------------------------------------------------------------------
            #Add the video Layouts to the MainLayout
            MainLayout.addLayout(VideoLayout)

            # -----------------------------------------------------------------------------------------------------
            #Defining a ScrollArea to enable the user to scroll up/down and left/right in the GUI
            #to prevent GUI content to be cut off at the bottom
            scrollable_window = QWidget()
            scrollable_window.setLayout(MainLayout)
            scroller = QScrollArea()
            scroller.setWidgetResizable(True)
            scroller.setWidget(scrollable_window)

            scroll_layout = QVBoxLayout(self)
            scroll_layout.addWidget(scroller)
            self.setLayout(scroll_layout)

            #Start Size of the Window
            #BUG WARNING: HORIZONTALLY RESIZING THE FRAME WILL MAKE THE SAMPLE DRAWING DISPLACED ABOVE A WINDOW WITH OF
            #1340PIXELS!!!
            self.resize(1340, 1080)
            self.setMaximumWidth(1340)
            # -----------------------------------------------------------------------------------------------------
            #Timer to refresh the video frames every 33 seconds (30fps)
            self.refreshGUI = QTimer(self)
            self.refreshGUI.timeout.connect(self.process_pipeline_signal)
            self.refreshGUI.start(33)
            #endregion

        #Error Handling
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in SetupGUI:", e)
                print(traceback.format_exc())
                error_reported = True
        # Functions to toggle visibility of sliders
    # -----------------------------------------------------------------------------------------------------
    #region ToggleSliders
    #toggle the Sliders whether the NLmeans_checkbox is checked
    def make_nlmneans_slider_visible(self, state):
            visible = state == Qt.Checked
            self.color_filter_label.setVisible(visible)
            self.color_filter_slider.setVisible(visible)
            self.luminance_filter_label.setVisible(visible)
            self.luminance_filter_slider.setVisible(visible)
            # Sets the bilateral_sliders visibale if the bilateral_filter_checkbox is checked

    def make_bilateral_sliders_visible(self, state):
        visible = state == Qt.Checked
        self.bilateral_kernel_size_label.setVisible(visible)
        self.bilateral_kernel_size.setVisible(visible)
        self.bilateral_sigma_color_label.setVisible(visible)
        self.bilateral_sigma_color.setVisible(visible)
        self.bilateral_sigma_space_label.setVisible(visible)
        self.bilateral_sigma_space.setVisible(visible)

    def make_cuda_bilateral_filter_sliders_visible(self, state):
        visible = state == Qt.Checked
        self.cuda_bilateral_kernel_size_label.setVisible(visible)
        self.cuda_bilateral_kernel_size.setVisible(visible)
        self.cuda_bilateral_color_label.setVisible(visible)
        self.cuda_bilateral_color.setVisible(visible)
        self.cuda_bilateral_space_label.setVisible(visible)
        self.cuda_bilateral_space.setVisible(visible)

    def make_multi_denoise_sliders_visible(self, state):
        visible = state == Qt.Checked
        self.multi_luminance_slider.setVisible(visible)
        self.multi_luminance_label.setVisible(visible)
        self.multi_color_slider.setVisible(visible)
        self.multi_color_label.setVisible(visible)


    # -----------------------------------------------------------------------------------------------------
    #updates the timing label in the GUI
    def update_timing_label(self, processing_times):
        try:
            text = "Processing Times: "
            fps = 0
            #For every function (key) show the processing time of this function (stored in processing_times)
            for function, time in processing_times.items():
                text += f"{function}: {time*1000:.0f} ms | "
                #add the processing time of the current function to the fps value
                fps = processing_times[function] + fps

            #concatenate the whole processing time string to the fps value string
            if fps != 0:
                text = f"<b>FPS:</b> {1/(fps):.0f} || " + text
            else:
                text = f"<b>FPS:</b> {0:.0f} || " + text

            #Set the label to the text
            self.timing_label.setText(text)
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in update_timing_label:", e)
                print(traceback.format_exc())
                error_reported = True

    #Drawing sample areas
    # -----------------------------------------------------------------------------------------------------
    #Defines the action taken when the user presses the left mouse button
    def mousePressEvent(self, pressing_left_mouse):
        try:
            #If the user is not drawing yet initialize empty lists for yuv-values
            if not hasattr(self, "draw_new_shape"):
                self.draw_new_shape = False
                self.y_values = []
                self.u_values = []
                self.v_values = []
                self.points = []

            #Set the drawing variable to true as soon as the left mouse button is clicked
            if self.original_label.underMouse():
                if pressing_left_mouse.button() == Qt.LeftButton:
                    self.draw_new_shape = True
                    # Reset the points list for the current drawing
                    self.points = []

        #Error Handling
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in mousePressEvent:", e)
                print(traceback.format_exc())
                error_reported = True


    #Defines the action taken when the user moves the mouse while keeping the left mouse button pressed
    def mouseMoveEvent(self, moving_mouse):
        try:
            #Convert the position to the coordinate system of the output video (left window)
            if self.draw_new_shape and self.original_label.underMouse():
                label_position = self.original_label.mapFromGlobal(moving_mouse.globalPos())
                x_label_coordinate = label_position.x()
                y_label_coordinate = label_position.y()

                #Get the width/height of the output video
                frame = getattr(self, 'current_frame')
                frame_height, frame_width,_ = frame.shape

                #Calculating the scale factors
                ScaleFactor_x = frame_width /self.original_label.width()
                ScaleFactor_y = frame_height /self.original_label.height()

                self.points.append((int(x_label_coordinate * ScaleFactor_x), int(y_label_coordinate * ScaleFactor_y)))

                #uses cv2.polylines to draw the lines on the original_frame output video
                overlay_drawing_frame = frame.copy()
                if len(self.points) > 1:
                    cv2.polylines(overlay_drawing_frame, [np.array(self.points)], False, (0, 255, 0), 2)

                #update the original_frame to show the current drawing
                self.display_original_frame(self.original_label, overlay_drawing_frame)

        #Error Handling
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in mouseMoveEvent:", e)
                print(traceback.format_exc())
                error_reported = True


    #Defines the action taken when the user releases the left mouse button after drawing a sample shape
    def mouseReleaseEvent(self, event):
        try:
            #draw_new_shape is set to false -> user is done drawing
            if self.draw_new_shape and event.button() == Qt.LeftButton:
                self.draw_new_shape = False

                #Get current Frame
                frame = getattr(self, 'current_frame', None)

                #Connect the start and the endpoint of the drawing to get a closed shape
                if len(self.points) > 1:
                    self.points.append(self.points[0])
                    self.drawn_sample_areas.append(self.points.copy())

                    #Used for sampling the Values in the closed shape
                    self.extract_sample_values(frame)

                    #reset the current drawn points to make space for the next user sample drawing
                    self.points.clear()
        #Error Handling
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in mouseReleaseEvent:", e)
                print(traceback.format_exc())


    #Extracts the YUV values from the closed shape
    def extract_sample_values(self, frame):
        try:
            #Create Mask with the frame shape
            shape_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            #Fills up the inside area of the closed shape and creates mask where
            #Mask = 255: point is inside the shape, Mask = 0: point is not inside the shape
            drawn_shape = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(shape_mask, [drawn_shape], 255)

            self.drawn_sample_areas.append(self.points.copy())  # Store completed shape

            #Get the pixel indices of all mask coordinates where the mask = 255 (inside shape)
            y_indices, x_indices = np.where(shape_mask == 255)

            # Prepare the new set of sampled points
            sample_points = {"y_values": [],"u_values": [],"v_values": []}

            #Extract the YUV-values from the pixels
            for y_indices, x_indices in zip(y_indices, x_indices):
                y, u, v = frame[y_indices, x_indices]
                sample_points["y_values"].append(y)
                sample_points["u_values"].append(u)
                sample_points["v_values"].append(v)

            #Add the sample set to the sample_set queue
            self.sample_sets.append(sample_points)

            #print the action taken and update the sample counter
            print(f"New sample set length with {len(sample_points['y_values'])} sample pixels. Total number of sample sets: {len(self.sample_sets)}")
            if self.counter < self.maximum_components_slider.value():
               self.counter += 1

        #Error Handling
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in extract_sample_values:", e)
                print(traceback.format_exc())
                error_reported = True


    #Retrieve the YUV values from the sample set for the GMM/BGMM function
    def get_yuv_values(self):
        try:
            y_values, u_values, v_values = [], [], []

            for sample_set in self.sample_sets:
                y_values.extend(sample_set["y_values"])
                u_values.extend(sample_set["u_values"])
                v_values.extend(sample_set["v_values"])

            return y_values, u_values, v_values

        #Error Handling
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in get_yuv_values:", e)
                print(traceback.format_exc())
                error_reported = True

    #Returns the counter value of the sample queue
    def get_counter(self):
        return self.counter

    # -----------------------------------------------------------------------------------------------------
    #Creates the YUV-Slider for Standard ChromaKeying
    def create_slider(self, label_text, min_value, max_value):
        label = QLabel(label_text)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_value, max_value)
        slider.setValue(min_value)

        #Add the Sliders/Label to the Layout
        self.chroma_sliders_layout.addWidget(label)
        self.chroma_sliders_layout.addWidget(slider)
        return slider

    # -----------------------------------------------------------------------------------------------------
    #Updates the threshold label
    def update_threshold_label(self):
        self.threshold_label.setText(f"Threshold: {self.threshold_slider.value()}")

    #Updates the Maximum Queue Lengths based on the slider value
    def update_maximum_queque_length(self):
        try:
            #Updates the maximum_components label
            maximum_components = self.maximum_components_slider.value()
            self.maximum_components_label.setText(f"Maximum Components: {maximum_components}")

            #Create new Queue with the specified slider value
            current_samples = list(self.sample_sets)
            self.sample_sets = deque(current_samples, maxlen=maximum_components)

        #Error Handling
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in update_maximum_queue_length:", e)
                print(traceback.format_exc())
                error_reported = True


    #Main update functions to pass to main and update the gui elements
    # -----------------------------------------------------------------------------------------------------
    #endregion
    #region Update/Display Frame

    #Connect to the main for refreshing the GUI.
    def process_pipeline_signal(self):
        self.process_pipeline()

    def update_label(self, slider, label, text_prefix):
        label.setText(f"{text_prefix}{slider.value()}")

    #Displays the processed_frame in the GUI
    def display_preprocessed_frame(self, label, frame):
        try:
            #Convert to RGB Colorspace to use for the QImage widget
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)

            #Get the frame dimensions of the frame
            height, width, interpret = frame.shape

            #Display the image in the Video Label
            processed_frame_image = QImage(frame.data, width, height, interpret * width, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(processed_frame_image))

        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in update_timing_label:", e)
                print(traceback.format_exc())
                error_reported = True

    #Displays the original_frame in the GUI
    def display_original_frame(self, label, frame):
        try:
            #Convert to RGB Colorspace to use for the QImage widget
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)

            #Get the frame dimensions of the frame
            height, width, interpret = frame.shape

            #Display the image in the Video Label
            original_frame_image = QImage(frame.data, width, height, interpret * width, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(original_frame_image))
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in display_original_frame:", e)
                print(traceback.format_exc())
                error_reported = True

    def display_preprocessed_frame(self, label, frame):
        try:
            # Convert to RGB Colorspace to use for the QImage widget
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)

            # Get the frame dimensions of the frame
            height, width, interpret = frame.shape

            # Display the image in the Video Label
            processed_frame_image = QImage(frame.data, width, height, interpret * width, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(processed_frame_image))
        except Exception as e:
            global error_reported
            if not error_reported:
                print("An error occured in display_preprocessed_frame:", e)
                print(traceback.format_exc())
                error_reported = True


