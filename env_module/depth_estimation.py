from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import pandas as pd
# from utils import str2list
checkpoint = "vinvino02/glpn-nyu"

image_processor = GLPNFeatureExtractor.from_pretrained(checkpoint)
model = GLPNForDepthEstimation.from_pretrained(checkpoint)

def estimate(im):
    image = Image.open(im)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    output = prediction.numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")

    # depth = Image.fromarray(formatted)
    depth_scale = 0.01
    return formatted*depth_scale
    #return depth.getpixel((x,y))*depth_scale

def estimate_evaluation(im,x,y):
    image = Image.open(im)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    output = prediction.numpy()
    # print value range
    # print(f'output shape {output.shape} max {np.max(output)} min {np.min(output)}')
    formatted = (output * 255 / np.max(output)).astype("uint8")
    return formatted[y][x]*0.01

def depth_estimation(ie_visual_dir, images_path):
    source_path_color = os.path.join(ie_visual_dir, "images", images_path, "color")
    png_files = [file for file in os.listdir(source_path_color) if file.endswith('.png')]
    output_path_depth = os.path.join(ie_visual_dir, "images", images_path, "depth")

    depth_matrices = []
    for png_file in sorted(png_files):
        print(f'file {png_file}')
        depth_array = estimate(os.path.join(source_path_color,png_file))
        depth_matrices.append(depth_array)

    np.save(os.path.join(output_path_depth,'depth_estimated'), np.array(depth_matrices))

def getDepthFromCamera(i,x_center,y_center):
    depth_matrices = np.load('/home/ziyu/semantic_programming/data/processed/images/134929/depth/depth.npy')
    return depth_matrices[i,y_center,x_center]

def convert_hand_bbox_to_xyz(bboxes, png_files,color_image_path):
    # i is the frame number
    counter = 0
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        if len(bbox) == 0:
            x = 0
            y = 0 
            z = 0
        else:
            x_center = math.floor(bbox[0] + (bbox[2] - bbox[0])/2)
            y_center = math.floor(bbox[1] + (bbox[3] - bbox[1])/2)
            im = os.path.join(color_image_path,png_files[i])
            print(f'hand center of x, y, {x_center} {y_center} camera z {getDepthFromCamera(i,x_center,y_center)} , estimated z {estimate_evaluation(im,x_center,y_center)}')





if __name__ == '__main__':
    # compare depth values from camera and estimation
    processed_path = "/home/ziyu/semantic_programming/data/processed"
    color_image_path = os.path.join(processed_path,"images/134929/color")
    png_files = [file for file in os.listdir(color_image_path) if file.endswith('.png')]
    # hand detection results
    file = os.path.join(processed_path,'images/134929','det.csv')
    df = pd.read_csv(file, header=None)
    print(f'length of dataframe {len(df)}')
    # select the third column with bbox [x1,y1,x2,y2] 
    df[2] = df[2].apply(str2list)
    # create hand trajectory in camera frame: num_frames x [xcenter, ycenter, zcenter]
    hand_traj = convert_hand_bbox_to_xyz(df[2],png_files,color_image_path)
    # object detection results