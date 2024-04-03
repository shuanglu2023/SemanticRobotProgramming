from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import math

# result T_cam_tool from simulation
# camera posiiton and orientation in tool coordinate
# assumption
qx1 = 0
qy1 = 0
qz1 = 0.707
qw1 = 0.707
x1 = 0
y1 = 0
z1 = -0.1


def euler2quat():
    r1 = R.from_euler('z', -90, degrees=True)
    print(r1.as_quat())


# result T_camlink_world in simulation
# camlink posiiton and orientation in world coordinate
qx = 0.7071
qy = 0.7071
qz = -0.00139857
qw = -0.001448
x = -0.498
y = -0.1
# table height 0.74
z = 0.596 + 0.74

def rotate(xy, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    return (
        xy[0] * cos_theta - xy[1] * sin_theta,
        xy[0] * sin_theta + xy[1] * cos_theta
    )


def translate(xy, offset):
    return xy[0] + offset[0], xy[1] + offset[1]

def quat_tfmatrix(x,y,z,qx,qy,qz,qw):
    T = np.zeros([4,4])
    T[0:3,0:3] = R.from_quat([qx,qy,qz,qw]).as_matrix()
    T[0,3] = x
    T[1,3] = y
    T[2,3] = z
    T[3,3] = 1
    return T

def euler_tfmatrix(x,y,z,rx,ry,rz):
    T = np.zeros([4,4])
    T[0:3,0:3] = R.from_euler('xyz',[rx,ry,rz],degrees=True).as_matrix()
    T[0,3] = x
    T[1,3] = y
    T[2,3] = z
    T[3,3] = 1
    return T

# camera in world coordinate
T_world_camlink = quat_tfmatrix(x,y,z,qx,qy,qz,qw)
T_camlink_cam =  quat_tfmatrix(x1,y1,z1,qx1,qy1,qz1,qw1)

def tf2rot(T):
    r = R.from_matrix(T[0:3,0:3])
    return r.as_matrix()

def tf2pos(T):
    p = T[0:3,3]
    return p

def tf2deg(T):
    r = R.from_matrix(T[0:3,0:3])
    return r.as_euler('xyz',degrees=True)


def get_pose(data):
    T_cam_obj = euler_tfmatrix(data[0],data[1],data[2],0,0,0)
    T_world_obj = T_world_camlink@T_camlink_cam@T_cam_obj
    # 3x3 rotation matrix
    r = tf2rot(T_world_obj)
    # euler angles
    d = tf2deg(T_world_obj)
    # position
    p = tf2pos(T_world_obj)

    pc = [p[0],p[1],p[2]]
    return pc


if __name__ == "__main__":
    # data = [0.058788, -0.042198, 0.779497]
    data = [0.016122, -0.553473, 0.779497]
    pos = get_pose(data)
    print(f'object position in base frame {pos}')