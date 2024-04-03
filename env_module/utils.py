import numpy as np
import configparser
import os
import math
import cv2
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.mixture import GaussianMixture
from scipy import stats

plt.rcParams.update({'font.size':36})


def pxpy2xyz():
    pass


def convert_strings_to_floats(x):
    y = x.replace('(',',')
    z = y.replace(')',',')
    result = z.split(',')
    value = []
    for r in result:
        try:
            value.append(float(r))
        except:
            pass
    return np.array(value)

def convert_list_to_floats(x):
    try:
        y = x.replace('[',',')
        z = y.replace(']',',')
        result = z.split(',')
        value = []
        for r in result:
            try:
                value.append(float(r))
            except:
                pass
    except:
        value = [0,0,width,height]
    return np.array(value)

def convert_landmarks_to_xyz(index, landmarks):
    # index: select depth image in matrices
    # landmarks: the landmark from color image with index 
    X = []
    Y = []
    Z = []
    for landmark in landmarks:
        # pixel coordinate in image
        if(landmark[0]==0 and landmark[1]==0 and landmark[2]==0):
            x = 0
            y = 0 
            z = 0
        else:
            px = min(math.floor(landmark[0]* width), width-1) 
            py = min(math.floor(landmark[1] * height), height-1)

            z = depth_matrices[index,py,px] 
            x = (px-ppx)/fx * z 
            y = (py-ppy)/fy * z
        X.append(x)
        Y.append(y)
        Z.append(z)
        nx = np.array(X)
        nx = nx.reshape(len(X),1)
        ny = np.array(Y)
        ny = ny.reshape(len(Y),1)
        nz = np.array(Z)
        nz = nz.reshape(len(Z),1)

    return np.concatenate((nx,ny,nz), axis=1)

def convert_bbox_to_xyz(index, bboxes):
    # convert bbox x_center and y_center to point
    # bbox format: normalized x_center, y_center, w, h 
    idx = 0
    results = []
    for bbox in bboxes:
        if bbox[0] == 0 and bbox[1] == 0:
            x = 0
            y = 0 
            z = 0
        else:
            x_center = min(math.floor(bbox[0]*width),width-1)
            y_center = min(math.floor(bbox[1]*height),height-1)
            z = depth_matrices[index,y_center,x_center] 
            x = (x_center-ppx)/fx * z 
            y = (y_center-ppy)/fy * z
        results.append({idx: [x,y,z]})
        idx = idx + 1
    return results

def convert_xywh_to_xyxy(bboxes):
    # Output bbox format:x_min, y_min, x_max, y_max
    # input bbox format: normalized x_center, y_center, w, h
    x_center = bboxes[0]
    y_center = bboxes[1]
    w = bboxes[2]
    h = bboxes[3]

    x_min = x_center - w/2
    y_min = y_center - h/2

    x_max = x_center + w/2
    y_max = y_center + h/2

    x_min = max(math.floor(x_min*width),0)
    y_min = max(math.floor(y_min*height),0)

    x_max = min(math.floor(x_max*width),width-1)
    y_max = min(math.floor(y_max*height),height-1)

    return [x_min, y_min, x_max, y_max]

# list are saved in dataframe as string
def str2list(value):
    # print(f'value {value}')
    try: 
        cleaned_string = value.strip("[]").split()
        list_data = [math.floor(float(value)) for value in cleaned_string]
    except:
        cleaned_string = value.replace('[', '').replace(']', '').split(' ')
        nested_list = [[math.floor(float(value)) for value in sublist.split()] for sublist in cleaned_string]
        list_data = []
        for x in nested_list:
            if len(x)>0:
                list_data.append(x[0])
        # print(f'list_data {list_data}')
    return list_data

def draw_bbox(im,bbox,label):
    """
    Draw bbox on image.
    
    Args:
    - im: image read by CV
    - bbox: [x1,y1,x2,y2]
    - label: name of class -> str

    Returns:    
    - im with bbox and label
    """
    color = (0,255,0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_color = (0, 0, 255)  # Red color
    text_thickness = 2

    cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2], bbox[3]), color)
    cv2.putText(im,label,(bbox[0],bbox[1]),font,font_scale,text_color,text_thickness)

    cv2.imshow('image',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def convert_obj_bbox_to_xyz(df):
    df = df.fillna(method='ffill')
    # print(df)
    results = []
    for index, row in df.iterrows():
        x_center = math.floor(row['X']*width)
        y_center = math.floor(row['Y']*height)
        # print(f'object center of x, y, camera z, estimated z {x_center} {y_center} {getDepthFromCamera(index,x_center,y_center)} {getDepthFromEstimation(index,x_center,y_center)}')
        z = getDepthFromCamera(index,x_center,y_center)
        if z == 0:
            z = getDepthFromEstimation(index,x_center,y_center)
        # print(f'depth value of object {z}')
        x = (x_center-ppx)/fx * z 
        y = (y_center-ppy)/fy * z
        results.append([x,y,z])
    return results

def mean_average_filter(data, window_size):
    filtered_data = []
    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1, :]
        filtered_data.append(np.mean(window, axis=0))
    return np.array(filtered_data)

def smooth_traj(input):
    """
    Args:
        input: lenght of video * [x,y,z]
    """ 
    observations = np.array(input)
    # Create the Kalman Filter
    kf = KalmanFilter(transition_matrices=np.eye(3), observation_matrices=np.eye(3))

    # Smooth the trajectory using Kalman filtering
    smoothed_states = kf.smooth(observations)

    # Extract the smoothed trajectory from the Kalman filter output
    smoothed_trajectory = smoothed_states[0]

    return smoothed_trajectory

def smooth_traj2(input):
    """
    Args:
        input: lenght of video * [x,y,z]
    """ 
    observations = np.array(input)

    window_size = 3
    filtered_trajectory = mean_average_filter(observations, window_size)

    return filtered_trajectory

def traj_speed(traj):
    """
    Args:
        input: lenght of video * [x,y,z]
    """
    diff = np.diff(traj, axis=0)
    distances = np.linalg.norm(diff,axis=1)
    time_intervals = 1/30
    speeds = distances/time_intervals
    return speeds

def traj_acc(speed):
    """
    Args:
        input: lenght of speed
    """
    diff = np.diff(speed)
    time_intervals = 1/30
    acc = diff/time_intervals
    return acc


def vis_hand_traj(hand_traj):
    hand_traj = np.array(hand_traj)
    # hand trajectory
    X = hand_traj[:,0]
    Y = hand_traj[:,1]
    Z = hand_traj[:,2]
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the lines
    ax.plot(range(0,len(X)), X, label='X')
    ax.plot(range(0,len(Y)), Y, label='Y')
    ax.plot(range(0,len(Z)), Z, label='Z')

    # Set labels and title
    ax.set_xlabel('timestamp')
    ax.set_ylabel('Position (mm)')
    ax.set_title('hand trajectory')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

def vis_hand_obj_traj(hand_traj,obj_traj,data):
    hand_traj = np.array(hand_traj)
    obj_traj = np.array(obj_traj)
   
    # hand trajectory
    X = hand_traj[:,0]
    Y = hand_traj[:,1]
    Z = hand_traj[:,2]

    # object trajectory
    OX = obj_traj[:,0]
    OY = obj_traj[:,1]
    OZ = obj_traj[:,2]

    # Create a figure and axis
    fig, (ax1,ax2) = plt.subplots(2,1)

    # Set labels and title
    ax1.set_xlabel('time stamp')
    ax1.set_ylabel('xyz')
    ax1.set_title('hand/object trajectory')

    # Add a legend
    ax1.legend()

    start,end=data[e]

    hand_speed = traj_speed(hand_traj)
    obj_speed = traj_speed(obj_traj)

    hand_speed[start-1] = 0
    hand_speed[end-1] = 0

    ax2.plot(range(0,len(hand_speed)), hand_speed, color='b',label='hand speed')
    ax2.plot(range(0,len(obj_speed)), obj_speed, color='y',label='object speed')
    x = [0.1]*len(obj_speed)
    ax2.plot(range(0,len(x)), x, color='r',label='threshold')
    ax2.set_xlabel('time in s')
    ax2.set_ylabel('speed m/s')
    ax2.legend()


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