import cv2
import time
from collections import deque
import traceback

buffer = deque(maxlen=7)

#Used to upload and initialize the parameters for the GPU
def prepare_GPU(frame):
    try:
        gpu_source = cv2.cuda.GpuMat()
        gpu_source.upload(frame)
        gpu_destination = cv2.cuda.GpuMat()
        return gpu_source, gpu_destination
    #Error Handling
    except Exception as e:
        print(f"An error occured in prepare_GPU: {e}")
        print(traceback.format_exc())
        return None, None

#Used to apply the specified Brightness/Contrast adjustments from the GUI-Sliders
def frame_brightness_contrast_adjustment(frame, brightness=0, contrast=0):
    #split the channels of the YUV-Frame
    y,u,v = cv2.split(frame)

    #Adjusts Brightness and contrast by scaling the pixel intensities (alpha) and adding the specified offest (beta)
    y_new = cv2.convertScaleAbs(y, alpha=1 + contrast/100.0, beta=brightness)

    #Merge the new y value and u,v back together
    return cv2.merge([y_new, u, v])

#Applies the specified noise_filter checked in the GUI
def filter_noise(app, frame):
    try:
        #Apply the NL-means filter (CUDA) to the processed_frame when the corresponding checkbox is selected
        if app.NLmeans_checkbox.isChecked():
            #Upload frame to GPU
            gpu_source, gpu_destination = prepare_GPU(frame)

            #Error handling
            if gpu_source is None or gpu_destination is None:
                raise Exception("CUDA could not be used")

            # src = input image for the GPU, dst = Output image container,
            # photo_render=filter strength color, h_luminance = filter strength for brightness
            frame = cv2.cuda.fastNlMeansDenoisingColored(src=gpu_source, dst=gpu_destination,
                                                         photo_render=app.color_filter_slider.value(),
                                                         h_luminance=app.luminance_filter_slider.value())

            #download the frame from the GPU back to the CPU
            frame = frame.download()

        #Apply the bilateral filter to the processed_frame when the corresponding checkbox is selected
        if app.bilateral_filter_checkbox.isChecked():
            # d = kernel size, sigmaColor = Filtering in Color space, sigmaSpace = Filtering in space domain
            frame = cv2.bilateralFilter(frame, d=app.bilateral_kernel_size.value(), sigmaColor=app.bilateral_sigma_color.value(), sigmaSpace=app.bilateral_sigma_space.value())

        #Apply the bilateral filter (CUDA) to the processed_frame when the corresponding checkbox is selected
        if app.cuda_bilateral_filter_checkbox.isChecked():
            gpu_source, gpu_destination = prepare_GPU(frame)

            #Error handling
            if gpu_source is None or gpu_destination is None:
                raise Exception("CUDA could not be used")

            # src = input image for the GPU, dst = Output image container,
            #kernel size = size of the kernel, sigma_color = Filter Strength for color, simga_spatial = filter strengh in spacial domain
            frame = cv2.cuda.bilateralFilter(src=gpu_source, dst=gpu_destination,
                                             kernel_size=app.cuda_bilateral_kernel_size.value(),
                                             sigma_color=app.cuda_bilateral_color.value(),
                                             sigma_spatial=app.cuda_bilateral_space.value())
            frame = frame.download()

        #Apply the temporal_denoising to the frame when the corresponding checkbox is selected
        if app.multiFrameDenoise_Checkbox.isChecked():

            #frame buffer to store the current frame
            buffer.append(frame)

            #If the length of the buffer is greater or equal the specified temporal_winder Value from the slider in the GUI
            #multi-frame denoising is applied
            temporalWindowSize = 7
            if len(buffer) >= temporalWindowSize:

                #central frame of the buffer
                middleframe = temporalWindowSize//2

                #Convert buffer to a list
                frame_buffer = list(buffer)

                #srcimgs = list of frames from the buffer, imgToDenoiseIndex = midframe: Central frame of buffer is denoised,
                #temporalWindowSize = number of frames used, hcolor= Filter Strength for color, h = filter strength for luminance
                #templatewindowsize = size of the window used to find similarities between similar regions
                #searchWindowSize = Window size where similar regions get searched at.
                frame = cv2.fastNlMeansDenoisingColoredMulti(srcImgs=frame_buffer, imgToDenoiseIndex=middleframe,
                                                             temporalWindowSize=7,
                                                             hColor=app.multi_color_slider.value(), h=app.multi_luminance_slider.value(), templateWindowSize=7,
                                                             searchWindowSize=7)
        return frame

    #Error Handling
    except Exception as e:
        print(f"An Error occured in filter_noise: {e}")
        print(traceback.format_exc())
        return frame, 0
