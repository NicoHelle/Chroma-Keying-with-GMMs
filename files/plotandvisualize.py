import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plot
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

#The Function takes the samples and the computed gmm or bgmm and plots the samples, the gmm/bgmm and its classification thresshold
def plot_ChromaKeying(yuv_samples, GMM_or_BGMM, window):
    try:
        print('Plotting GMM/BGMM Chromakeying samples')
        #initialize figure
        figure = plot.figure()
        axis = figure.add_subplot(111, projection='3d')

        #plot the yuv_samples
        axis.scatter(yuv_samples[:, 0], yuv_samples[:, 1], yuv_samples[:, 2] ,label='Samples', c='green', alpha=0.75, marker='x')

        #Takes the threshold value from the GUI slider
        threshold = window.threshold_slider.value()

        #The Function plots the classification threshold and the GMM-2 sigma boundary
        def plot_boundaries(gmm, axis, color, threshold_ellipse=False):
            for mean, covariance in zip(gmm.means_, gmm.covariances_):
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)

                if not threshold_ellipse:
                    #take the 2-sigma boundary (double standard deviation)
                    radius = 2 * np.sqrt(eigenvalues)

                #Compute the boundary
                else:
                    dimension = 3
                    determinant = np.linalg.det(covariance)
                    constant = dimension * np.log(2 * np.pi) + np.log(determinant)
                    r2 = -2 * threshold - constant

                    # The radius gets scaled by sqrt(eigenvalue)
                    radius = np.sqrt(r2 * eigenvalues)

                # Create the ellipsoid
                u = np.linspace(0, 2 * np.pi, 50)
                v = np.linspace(0, np.pi, 50)

                x_position = radius[0] * np.outer(np.cos(u), np.sin(v))
                y_position = radius[1] * np.outer(np.sin(u), np.sin(v))
                z_position = radius[2] * np.outer(np.ones_like(u), np.cos(v))

                #Rotate and translate the calculated ellipsoid
                for i in range(len(x_position)):
                    for j in range(len(x_position)):
                        [x_position[i, j], y_position[i, j], z_position[i, j]] = np.dot(eigenvectors, [x_position[i, j], y_position[i, j], z_position[i, j]]) + mean

                axis.plot_wireframe(x_position, y_position, z_position, color=color, alpha=0.1)

        #Draw the threshold ellipsoid (red)
        plot_boundaries(GMM_or_BGMM, axis, 'red', threshold_ellipse=True)
        # Draw the standard ellipsoid(green))
        plot_boundaries(GMM_or_BGMM, axis, 'green')

        #Label the axes
        axis.set_xlabel('Y')
        axis.set_ylabel('U')
        axis.set_zlabel('V')
        axis.legend(['Samples', 'GMM', 'Classification Threshold'])
        plot.show()
    except Exception as e:
            print("An Error occured in plot_ChromaKeying:", e)
            print(traceback.format_exc())

