from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
## segmentation the trajectories
def segmentation(hand_traj,obj_traj,visualize=True):

    # object trajectory
    OX = obj_traj[:,0]
    OY = obj_traj[:,1]
    OZ = obj_traj[:,2]

    hand_traj = np.array(hand_traj)
    obj_traj = np.array(obj_traj)

    data_gmm = np.array(obj_traj)
    n_components = 3

    # Fit the Gaussian Mixture Model to the data
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data_gmm)

    # Get the final cluster labels for the data points
    labels = gmm.predict(data_gmm)
    zero_index = np.where(labels==labels[-1])
    # print('index',zero_index[0][0])

    if visualize:
        # Get the means and covariances of the Gaussian components
        means = gmm.means_
        # covariances = gmm.covariances_

        # Create a figure and axis
        fig, ax1 = plt.subplots()
        ax1.scatter(range(0,len(OX)), OX, c=labels, cmap='viridis')
        ax1.scatter(range(0,len(OY)), OY, c=labels, cmap='viridis')
        ax1.scatter(range(0,len(OZ)), OZ, c=labels, cmap='viridis')
        # ax1.text(0, 1, 'GMM 1', fontsize=36, horizontalalignment='left', verticalalignment='center')

    return zero_index[0][0], ax1