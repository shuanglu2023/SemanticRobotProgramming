from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
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