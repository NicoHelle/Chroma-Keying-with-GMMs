import cv2
import numpy as np
from multipledispatch import dispatch
from sklearn.mixture import GaussianMixture
from plotandvisualize import plot_ChromaKeying
global mask
from main import update_mask
import time
from preprocess import *
from postprocess import *
import traceback
from sklearn.mixture import GaussianMixture
import math
import cupy as cp
from sklearn.mixture import BayesianGaussianMixture
import os

error_reported = False #Used to not print duplicate error messages for functions in the main pipeline

#Computes the log-sum-exp for the axis, needed to compute log-likelihoods without rounding errors
def logsumexp(values, axis=1):
    try:
        #subtract the maximum
        maximum_values = cp.max(values, keepdims=True, axis=axis)

        #sum the exponent
        exponent_sum = cp.sum(cp.exp(values - maximum_values), keepdims=True, axis=axis)

        #take the log of the sum and then add back max_values
        return (maximum_values + cp.log(exponent_sum)).squeeze(axis=axis)
    except Exception as e:
        print("An error occured in logsumexp", e)
        print(traceback.format_exc())

#Compute the log_likelihoods with the GPU
def compute_log_likelihoods(frame, means, covariances, weights):
    try:
        pixel_number = frame.shape[0]
        components_number = means.shape[0]

        #Store the log_probabilitys for each component
        log_probabilities = cp.zeros((pixel_number, components_number), dtype=cp.float64)

        #Computes for each component the log-likelihood that a sample belongs to that component
        for i in range(components_number):
            weight = weights[i]
            mean = means[i]
            covariance = covariances[i]

            #Compute the determinant of the covariance matrix and the inverse
            determinant = cp.linalg.det(covariance)
            inverse_covariance = cp.linalg.inv(covariance)

            diffrence = frame - mean

            #Compute the Mahalanobis distance for each sample
            mahalanobis_distance = cp.einsum('ij,jk,ik->i', diffrence, inverse_covariance, diffrence)

            #log of determinant
            log_determinant = cp.log(determinant)

            #Compute the constant part for the current component
            log_constant = cp.log(weight) - 1.5*cp.log(2.0*cp.pi) - 0.5*log_determinant

            #Commpute the result
            log_probabilities[:, i] = log_constant - 0.5 * mahalanobis_distance

        # Use the logsumexp function to combine the log-likelihoods of the components
        log_likelihoods = logsumexp(log_probabilities, axis=1)

        return log_likelihoods

    except Exception as e:
        print("An error occured in compute_log_likelihoods", e)
        print(traceback.format_exc())


#Computes the chroma keying mask using a Bayesian Gaussian Mixture Model (BGMM)
def BGMM_ChromaKeying(window, frame, original_frame, new_samples=False):
    try:
        #Retrieve samples when a new sample area has been drawn
        if new_samples:
            y_values, u_values, v_values = window.get_yuv_values()

            #Returns original_frame when there are no samples available yet
            if len(y_values) == 0:
                print("No YUV values provided for BGMM computation.")
                return original_frame

            #Restructure yuv values
            yuv_samples = np.array([y_values, u_values, v_values]).T

            #Retrieve the Slider Value from the GUI for the maximum components
            components_number = window.maximum_components_slider.value()

            #Fit the BGMM model with the number of components. The random_state, weight_concentration_prior, weight_concentration_prior_type are optional parameters
            BGMM = BayesianGaussianMixture(n_components=components_number,
                                           covariance_type='full', random_state=1,
                                           weight_concentration_prior=0.1,
                                           weight_concentration_prior_type="dirichlet_process")
            computed_BGMM = BGMM.fit(yuv_samples)

            #Optional: print BGMM weights to compare with GMM weights
            print("BGMM Weights:", computed_BGMM.weights_)

            #store the trained gmm in the GUI
            window.gmm_fitted = computed_BGMM

            #Plot the Samples and the computed BGMM
            plot_ChromaKeying(yuv_samples, computed_BGMM, window)
            return

        #Return original frame when BGMM is not computed yet
        if not hasattr(window, 'gmm_fitted'):
            return original_frame

        computed_BGMM = window.gmm_fitted

        #Retrieves the threshold value
        threshold = window.threshold_slider.value()

        #Use float64 to avaoid rounding errors
        frame_float = frame.reshape(-1, 3).astype(np.float64)

        #Copy the BGMM parameters to the GPU for log-likelihood computation
        means = cp.asarray(computed_BGMM.means_, dtype=cp.float64)
        covariances = cp.asarray(computed_BGMM.covariances_, dtype=cp.float64)
        weights = cp.asarray(computed_BGMM.weights_, dtype=cp.float64)
        frame_gpu = cp.asarray(frame_float)

        #Compute log-likelihoods on the GPU
        log_likelihoods_gpu = compute_log_likelihoods(frame_gpu, means, covariances, weights)

        #Get computed log-likelihoods to CPU
        log_likelihoods = log_likelihoods_gpu.get()

        # Create the binay mask based on if the log-likelihood of a pixel is smaller (or equal) or greater than the threshold.
        # if log-likelihood < threshold => forgeground else background
        mask = (log_likelihoods < threshold).astype(np.uint8)
        mask = mask.reshape(frame.shape[:2])
        mask = (mask * 255).astype(np.uint8)

        #Compute the Alpha Matte of the mask with a dedicated function.
        result = compute_alpha_matte(original_frame, mask, window)
        return result

    #Exception Handling
    except Exception as e:
        global error_reported
        if not error_reported:
            print("An Error occured in BGMM_ChromaKeying:", e)
            print(traceback.format_exc())
            error_reported = True
        return original_frame

#Computes the chroma keying mask using a Gaussian Mixture Model (GMM)
def GMM_ChromaKeying(window, frame, original_frame, new_samples=False):
    try:
        #Retrieve samples when a new sample area has been drawn
        if new_samples:
            y_values, u_values, v_values = window.get_yuv_values()
            #Returns original_frame when there are no samples available yet
            if len(y_values) == 0:
                print("No YUV values provided for GMM computation")
                return original_frame

            #Restructure yuv values
            yuv_samples = np.array([y_values, u_values, v_values]).T

            # Retrieve the Slider Value from the GUI for the maximum components
            components_number = window.maximum_components_slider.value()

            # Fit the GMM model with the number of components. The random_state is an optional parameter.
            GMM = GaussianMixture(n_components=components_number, covariance_type='full', random_state=1)
            computed_GMM = GMM.fit(yuv_samples)

            #Optional: print GMM weights to compare with BGMM weights
            print("Standard GMM Weights:", computed_GMM.weights_)

            #store the trained gmm in the GUI
            window.gmm_fitted = computed_GMM  # store trained model in the GUI

            #Plot the Samples and the computed GMM
            plot_ChromaKeying(yuv_samples, computed_GMM, window)
            return

        # Return original frame when GMM is not computed yet
        if not hasattr(window, 'gmm_fitted'):
            return original_frame

        computed_GMM = window.gmm_fitted

        #Retrieves the threshold value
        threshold = window.threshold_slider.value()

        #Use float64 to avaoid rounding errors
        frame_float64 = frame.reshape(-1, 3).astype(np.float64)

        #Copy the GMM parameters to the GPU for log-likelihood computation
        means = cp.asarray(computed_GMM.means_, dtype=cp.float64)
        covariances = cp.asarray(computed_GMM.covariances_, dtype=cp.float64)
        weights = cp.asarray(computed_GMM.weights_, dtype=cp.float64)
        frame_gpu = cp.asarray(frame_float64)

        #Compute log-likelihoods on the GPU
        log_likelihoods_gpu = compute_log_likelihoods(frame_gpu, means, covariances, weights)
        log_likelihoods = log_likelihoods_gpu.get()  # copy back to CPU

        #Create the binay mask based on if the log-likelihood of a pixel is smaller (or equal) or greater than the threshold.
        # if log-likelihood < threshold => forgeground else background
        mask = (log_likelihoods < threshold).astype(np.uint8)
        mask = mask.reshape(frame.shape[:2])
        mask = (mask * 255).astype(np.uint8)

        #Compute the Alpha Matte of the mask with a dedicated function.
        result = compute_alpha_matte(original_frame, mask, window)
        return result

    #Error Handling
    except Exception as e:
        global error_reported
        if not error_reported:
            print("An Error occured in GMM_ChromaKeying:", e)
            print(traceback.format_exc())
            error_reported = True
        return original_frame


#GMM computation ersion withtout GPU Computation: This Version is way slower than GMM_Chromakeying but only requires the CPU
def GMM_ChromaKeying_CPU(window, frame, original_frame, new_samples=False):
    try:
        # Retrieve samples when a new sample area has been drawn
        if new_samples:
            y_values, u_values, v_values = window.get_yuv_values()
            # Returns original_frame when there are no samples available yet
            if len(y_values) == 0:
                print("No YUV values provided for GMM computation")
                return original_frame

            #Restructure yuv values
            yuv_samples = np.array([y_values, u_values, v_values]).T

            # Retrieve the Slider Value from the GUI for the maximum components
            components_number = window.maximum_components_slider.value()

            # Fit the GMM model with the number of components. The random_state is an optional parameter.
            GMM = GaussianMixture(n_components=components_number, covariance_type='full', random_state=1)
            computed_GMM = GMM.fit(yuv_samples)

            #Optional: print GMM weights to compare with BGMM weights
            print("Standard GMM Weights:", computed_GMM.weights_)

            #store the trained gmm in the GUI
            window.gmm_fitted = computed_GMM  # store trained model in the GUI

            #Plot the Samples and the computed GMM
            plot_ChromaKeying(yuv_samples, computed_GMM, window)
            return

        # Return original frame when GMM is not computed yet
        if not hasattr(window, 'gmm_fitted'):
            return original_frame

        computed_GMM = window.gmm_fitted

        #Retrieves the threshold value
        threshold = window.threshold_slider.value()

        #Use float64 to avaoid rounding errors
        frame_float64 = frame.reshape(-1, 3).astype(np.float64)

        #Uses the CPU to compute the Log-Likelihoods of the pixels with the score_samples built-in function
        log_likelihoods = computed_GMM.score_samples(frame_float64)

        #Create the binay mask based on if the log-likelihood of a pixel is smaller (or equal) or greater than the threshold.
        # if log-likelihood < threshold => forgeground else background
        mask = (log_likelihoods < threshold).astype(np.uint8)
        mask = mask.reshape(frame.shape[:2])
        mask = (mask * 255).astype(np.uint8)

        #Compute the Alpha Matte of the mask with a dedicated function.
        result = compute_alpha_matte(original_frame, mask, window)
        return result

    #Error Handling
    except Exception as e:
        global error_reported
        if not error_reported:
            print("An Error occured in GMM_ChromaKeying_CPU:", e)
            print(traceback.format_exc())
            error_reported = True
        return original_frame

#Uses the Slider values of the GUI to generate a mask and Chroma Key the image
def Standard_ChromaKeying(window, frame, original_frame):

    try:
        #Extract the slider Values of the GUI
        y_minimum_range = window.y_min_slider.value()
        y_maximum_range = window.y_max_slider.value()
        u_minimum_range = window.u_min_slider.value()
        u_maximum_range = window.u_max_slider.value()
        v_minimum_range = window.v_min_slider.value()
        v_maximum_range = window.v_max_slider.value()

        #When a maximum < minimum then return original_frame to save computation time in that specific case
        if y_minimum_range >= y_maximum_range or u_minimum_range >= u_maximum_range or v_minimum_range >= v_maximum_range:
            return original_frame

        #Computes the lower and upper bound of the y,u and v value
        lower_boundary = np.array([y_minimum_range, u_minimum_range, v_minimum_range], dtype=np.uint8)
        upper_boundary = np.array([y_maximum_range, u_maximum_range, v_maximum_range], dtype=np.uint8)

        #Computes the binary mask and invert it -> Foreground = 1, Background = 0
        mask = cv2.inRange(frame, lower_boundary, upper_boundary)
        mask = cv2.bitwise_not(mask) / 255.0

        #Alhpa Blending: detect edges of the inverted mask
        edge_image = cv2.Canny((mask * 255).astype(np.uint8), 25, 200)

        #Alpha Blending: Blur the edges of the edge image to get smoother transitions between background and foreground
        blur_edge_image = cv2.GaussianBlur(edge_image, (9, 9), 0) / 255.0

        #Add the original matte and the blur_edge_image together to get an alpha mask
        final_mask = np.clip(mask + blur_edge_image, 0, 1)
        final_mask = final_mask.astype(np.float32)

        #Optional in GUI: Apply Erosion to mask
        if window.use_erosion.isChecked():
            erosion_mask = np.ones((window.erosion_slider.value(), window.erosion_slider.value()), np.uint8)
            final_mask = cv2.erode(final_mask, erosion_mask)

        #Optional in GUI: Apply dilation to mask
        if window.use_dilation.isChecked():
            dilation_mask = np.ones((window.dilation_slider.value(), window.dilation_slider.value()), np.uint8)
            final_mask = cv2.dilate(final_mask, dilation_mask)

        #Blend foreground and a custom background image
        background = os.path.join(os.path.dirname(__file__), 'test.jpg')
        background = cv2.cvtColor(cv2.imread(background), cv2.COLOR_BGR2YUV)
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        foreground = original_frame

        #Alpha blending result = foreground * final_mask + background * (1 - final_mask)
        result = (foreground * final_mask[:, :, None] + background * (1 - final_mask[:, :, None])).astype(np.uint8)
        return result

    except Exception as e:
        global error_reported
        if not error_reported:
            print("An error occured in Standard_ChromaKeying:", e)
            print(traceback.format_exc())
            error_reported = True
        return original_frame


#Returns the Mask, imported for visualizing samples of Standard_ChromaKeying.
def return_mask_chromakey(window, frame, original_frame):
    try:
        #Extract the slider Values of the GUI
        y_minimum_range = window.y_min_slider.value()
        y_maximum_range = window.y_max_slider.value()
        u_minimum_range = window.u_min_slider.value()
        u_maximum_range = window.u_max_slider.value()
        v_minimum_range = window.v_min_slider.value()
        v_maximum_range = window.v_max_slider.value()

        #When a maximum < minimum then return original_frame to save computation time in that specific case
        if y_minimum_range >= y_maximum_range or u_minimum_range >= u_maximum_range or v_minimum_range >= v_maximum_range:
            return original_frame

        #Computes the lower and upper bound of the y,u and v value
        lower_boundary = np.array([y_minimum_range, u_minimum_range, v_minimum_range], dtype=np.uint8)
        upper_boundary = np.array([y_maximum_range, u_maximum_range, v_maximum_range], dtype=np.uint8)

        #Computes the binary mask and invert it -> Foreground = 1, Background = 0
        mask = cv2.inRange(frame, lower_boundary, upper_boundary)
        mask = cv2.bitwise_not(mask) / 255.0

        #Alhpa Blending: detect edges of the inverted mask
        edge_image = cv2.Canny((mask * 255).astype(np.uint8), 25, 200)

        #Alpha Blending: Blur the edges of the edge image to get smoother transitions between background and foreground
        blur_edge_image = cv2.GaussianBlur(edge_image, (9, 9), 0) / 255.0

        #Add the original matte and the blur_edge_image together to get an alpha mask
        final_mask = np.clip(mask + blur_edge_image, 0, 1)
        final_mask = final_mask.astype(np.float32)

        #Optional in GUI: Apply Erosion to mask
        if window.use_erosion.isChecked():
            erosion_mask = np.ones((window.erosion_slider.value(), window.erosion_slider.value()), np.uint8)
            final_mask = cv2.erode(final_mask, erosion_mask)

        #Optional in GUI: Apply dilation to mask
        if window.use_dilation.isChecked():
            dilation_mask = np.ones((window.dilation_slider.value(), window.dilation_slider.value()), np.uint8)
            final_mask = cv2.dilate(final_mask, dilation_mask)

        return final_mask

    except Exception as e:
        global error_reported
        if not error_reported:
            print("An error occured in return_mask_chromakey:", e)
            print(traceback.format_exc())
            error_reported = True
        return original_frame





