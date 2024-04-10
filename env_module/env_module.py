from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.engine import DefaultPredictor
import os
import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import pandas as pd
from env_module.utils import str2list, smooth_traj, vis_hand_traj
from env_module.depth_estimation import depth_estimation
import math
import configparser
from env_module.yolov8.predict import output_obb
from task_module.entityDAO import TaskModelDAO, ProductDAO, ProductPositionDAO, GroundedProductDAO, HandPositionDAO
from env_module.segmentation import segmentation
import matplotlib.pyplot as plt
from db.database import create_product_positions_table, create_grounded_products_table
from task_module.entity import Product

class EnvModule():
    def __init__(self,task_model_id,db_path) -> None:
        # Get the directory of the current script
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.parent_directory = os.path.dirname(self.script_dir)
        self.raw_data_path = os.path.join(self.parent_directory,'data/raw')
        self.processed_path = os.path.join(self.parent_directory,'data/processed')
        self.read_camera_parameter()
        self.task_model_id = task_model_id
        self.db_path = os.path.join(self.parent_directory,db_path)
        # 1. Cuboid 2. Star 3. Parallelogram 4. Octangon , 3, 5, 2, 4 
        self.class_labels = ["graybox", "bluebox", "parallelogram", "cuboid", "octagon", "star"]
        # Creating label2id
        self.label2id = {label: id for id, label in enumerate(self.class_labels)}
        self.id2label = {id: label for label, id in self.label2id.items()}

    def extract_from_bag(self, bag_fname, color_fname, depth_fname):
        # https://github.com/IntelRealSense/librealsense/issues/4934
        # the final images are generated with this script without frame lost
        print(bag_fname, color_fname, depth_fname)
        config = rs.config()
        pipeline = rs.pipeline()

        # make it so the stream does not continue looping
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        rs.config.enable_device_from_file(config, bag_fname, repeat_playback=False)
        profile = pipeline.start(config)
        # this makes it so no frames are dropped while writing video
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        colorizer = rs.colorizer()

        align_to = rs.stream.color
        align = rs.align(align_to)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f'depth scale {depth_scale}')

        depth_matrices = []

        i = 0
        while True:

            # when stream is finished, RuntimeError is raised, hence this
            # exception block to capture this
            try:
                # frames = pipeline.wait_for_frames()
                frames = pipeline.wait_for_frames(timeout_ms=300)
                if frames.size() <2:
                    # Inputs are not ready yet
                    continue
            except (RuntimeError):
                print('frame count', i-1)
                pipeline.stop()
                break

            # align the deph to color frame
            aligned_frames = align.process(frames)

            # get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            scaled_depth_image = depth_image * depth_scale
            color_image = np.asanyarray(color_frame.get_data())

            # convert color image to BGR for OpenCV
            r, g, b = cv2.split(color_image)
            color_image = cv2.merge((b, g, r))

            depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow('Aligned Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Aligned Example', images)

            fname = "frame{:06d}".format(i) + ".png"
            cv2.imwrite(color_fname + fname, color_image)
            depth_matrices.append(scaled_depth_image)

            # color_out.write(color_image)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

            i += 1

        # release everything now that job finished
        # save depth map as matrices
        np.save(depth_fname, np.array(depth_matrices))
        print("Size of depth matrices:", len(depth_matrices))
        cv2.destroyAllWindows()

    def read_camera_parameter(self):
        # Create a ConfigParser object
        config = configparser.ConfigParser()
        # Read the configuration file
        config.read(os.path.join(self.script_dir,'config.ini'))
        self.width = int(config.get("ColorCameraIntrinsics", "width"))
        self.height = int(config.get("ColorCameraIntrinsics", "height"))
        self.fx = np.float16(config.get("ColorCameraIntrinsics", "fx"))
        self.fy = np.float16(config.get("ColorCameraIntrinsics", "fy"))
        self.ppx = np.float16(config.get("ColorCameraIntrinsics", "ppx"))
        self.ppy = np.float16(config.get("ColorCameraIntrinsics", "ppy"))

    def grounding_products(self):
        create_grounded_products_table(self.db_path)
        product_dao = ProductDAO(self.db_path)
        grounded_product_dao = GroundedProductDAO(self.db_path)
        products = product_dao.get_all_products()
        class_labels = []
        for product in products:
            product_name = product[1]
            product_color = product[4]
            class_labels.append({'color':product_color,'name':product_name})
        # Generating a list of unique names
        unique_names = list(set(item['name'] for item in class_labels))
        for product_name in unique_names:
            results = product_dao.get_products_by_name(product_name)
            for result in results:
                product_color = result[4]
                product_name = result[1]
                product_id = result[0]
                if product_color is not None:
                    product_name = product_color + product_name
                try:
                    grounded_product_dao.add_grounded_product(self.label2id[product_name],product_name)
                except:
                    pass
                source_location_id = result[2]
                target_location_id = result[3]
                if source_location_id is not None:
                    grounded_product_dao.update_source_location(self.label2id[product_name], source_location_id)
                if target_location_id is not None:
                    grounded_product_dao.update_target_location(self.label2id[product_name], target_location_id)
                grounded_product_dao.update_color(self.label2id[product_name], product_color)
                product_dao.add_class_id(self.label2id[product_name],product_id)

    def process_video(self,video_path):
        bag_file = video_path.split('/')[3]
        images_path = self.video2frame(bag_file)
        # hand detection and generate hand trajectories
        # depth_estimation(self.processed_path,images_path)
        # self.det_hand(images_path)
        self.depth_matrices = np.load(os.path.join(self.processed_path,'images',images_path,'depth','depth.npy'))
        self.depth_estimated = np.load(os.path.join(self.processed_path,'images',images_path,'depth','depth_estimated.npy'))
        hand_traj = self.process_hand(images_path)
        hand_traj = smooth_traj(hand_traj)
        # save hand_traj to database
        hand_position_dao = HandPositionDAO(self.db_path,self.task_model_id)
        hand_position_dao.add_hand_positions(hand_traj)
        # vis_hand_traj(hand_traj)
        path = os.path.join(self.processed_path,'images',images_path,'color')
        # object detection with yolov8 and save the data into csv files
        # self.det_objs(path,images_path,self.width,self.height)
        # lead obb data from csv files and convert to xyzr, save them into database
        self.grounding_products()
        self.process_objects(images_path)
        # segmentation the trajectories, save the grasp and release index into the database
        self.seg(hand_traj)

    def get_obj_sequences(self):
        task_model_dao = TaskModelDAO(self.db_path)
        products = task_model_dao.get_target_objects_for_task_model(self.task_model_id)
        product_dao = ProductDAO(self.db_path)
        results = []
        for product in products:
            results.append(self.label2id[product_dao.get_product(product)[1]])
        unique_list = [results[i] for i in range(len(results)) if i == 0 or results[i] != results[i-1]]
        return unique_list

    def read_product_position_from_db(self,product_name):
        product_position_dao = ProductPositionDAO(self.db_path,product_name)
        obj_traj = []
        results = product_position_dao.get_product_positions()
        for result in results:
            # print(f'result {result}')
            obj_traj.append([result[5],result[6],result[7]])
        return obj_traj

    def seg(self,hand_traj):
        # the sequences of moved objects during demonstration
        start = self.start
        end = self.end
        for i in self.get_obj_sequences():
            # read position from database
            obj_traj = self.read_product_position_from_db(self.id2label[i])
            obj_traj = smooth_traj(obj_traj)
            release_index, ax = segmentation(hand_traj,obj_traj,visualize=True)
            initial_position = np.mean(obj_traj[:start], axis=0)
            final_position = np.mean(obj_traj[end:], axis=0)
            print(f'object {self.id2label[i]} initial_position {initial_position} final_position {final_position} release_index {release_index}')
            dists = []

            for x in hand_traj[start:release_index]:
                dist = np.linalg.norm(x - initial_position)
                dists.append(dist)

            min_value = min(dists)
            min_index = dists.index(min_value)
            grasp_index = min_index + start

            ax.set_ylim(-0.3,0.7)
            ax.axvline(x=grasp_index)
            ax.annotate('grasp', xy=(grasp_index, 0.65), xytext=(grasp_index-20, 0.8), arrowprops=dict(arrowstyle='->'))
            ax.axvline(x=release_index)
            ax.annotate('release', xy=(release_index, 0.65), xytext=(release_index+0.1, 0.8), arrowprops=dict(arrowstyle='->'))
            print(f'grasp index {grasp_index}, release index {release_index}')
            # add grasp index and release index into the database, product positions table
            product_position_dao = ProductPositionDAO(self.db_path,self.id2label[i])
            product_position_dao.update_reach_index(int(start),int(grasp_index))
            product_position_dao.update_grasp_index(int(grasp_index))
            product_position_dao.update_move_index(int(grasp_index),int(release_index))
            product_position_dao.update_release_index(int(release_index))
            # ax.legend()
            plt.xlabel('timestamp')
            plt.ylabel('object position (m)')
            # plt.legend()
            # plt.show()
            start = release_index


    def generate_data_for_object_detection(self):
        pass

    def train_classifier_for_object_detection(self):
        pass

    def video2frame(self,bag_file):
        # extract the color and depth frames from video
        images_path = bag_file.split('.')[0].split('_')[1]
        bag_file_path = os.path.join(self.raw_data_path,bag_file)
        # self.extract_from_bag(bag_file_path, color_fname=os.path.join(self.processed_path,'images/'+images_path+'/color/'), depth_fname=os.path.join(os.path.join(self.processed_path,'images/'+images_path+'/depth/depth')))
        return images_path

    def det_hand(self,images_path):
        # run hand detection over all RGB images
        cfg = get_cfg()
        file_path = os.path.join(self.script_dir,'hand_detector.d2','faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml')
        cfg.merge_from_file(file_path)
        model_path = os.path.join(self.script_dir,'hand_detector.d2','models/model_0529999.pth')
        cfg.MODEL.WEIGHTS = model_path # add model weight here
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
        # predict
        predictor = DefaultPredictor(cfg)

        with open(os.path.join(self.processed_path,'images', images_path,'det.csv'),'w',newline='') as file:
            image_dir = os.path.join(self.processed_path,'images', images_path,'color')
            images = sorted(os.listdir(image_dir))
            for image in images:
                file_path = os.path.join(image_dir,image)
                im = cv2.imread(file_path)
                outputs = predictor(im)
                data = [os.path.join(image_dir,image),outputs["instances"].pred_classes.cpu().numpy(),outputs["instances"].pred_boxes.tensor.cpu().numpy()]
                writer = csv.writer(file)
                writer.writerow(data)

    def det_objs(self,path,images_path,width,height):
        output_obb(path,images_path,width,height,self.class_labels)


    def getDepthFromCamera(self,i,x_center,y_center):
        return self.depth_matrices[i,y_center,x_center]

    def getDepthFromEstimation(self,i,x_center,y_center):
        return self.depth_estimated[i,y_center,x_center]

    def convert_hand_bbox_to_xyz(self,bboxes):
        """
        Generate hand trajectories for a video
        
        Args:
        - im: image read by CV
        - bbox: [x1,y1,x2,y2]
        - label: name of class -> str

        Returns:    
        - im with bbox and label
        """
        results = []
        # print(f'size of bboxes {len(bboxes)}')
        start = 0
        end = len(bboxes)-1
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            # print(f'bbox {bbox}')
            if len(bbox) == 0:
                x = 0
                y = 0 
                z = 0
                if start > 0 and end == (len(bboxes)-1):
                    end = i
            else:
                if start==0:
                    start = i
                x_center = math.floor(bbox[0] + (bbox[2] - bbox[0])/2)
                y_center = math.floor(bbox[1] + (bbox[3] - bbox[1])/2)
                print(f'hand center of x, y, camera z, estimated z {x_center} {y_center} {self.getDepthFromCamera(i,x_center,y_center)} {self.getDepthFromEstimation(i,x_center,y_center)}')
                z = self.getDepthFromCamera(i,x_center,y_center)
                if z == 0:
                    z = self.getDepthFromEstimation(i,x_center,y_center)
                x = (x_center-self.ppx)/self.fx * z 
                y = (y_center-self.ppy)/self.fy * z
            results.append([x,y,z])
        self.start = start
        self.end = end
        return results

    def convert_obj_bbox_to_xyz(self,df):
        xyzr = []
        for index, row in df.iterrows():
            # print(f'{index} {row}')
            x_center = math.floor(row['X']*self.width)
            y_center = math.floor(row['Y']*self.height)
            rot = row['R']
            print(f'object center of x, y, camera z, estimated z {x_center} {y_center} {self.getDepthFromCamera(index,x_center,y_center)} {self.getDepthFromEstimation(index,x_center,y_center)}')
            z = self.getDepthFromCamera(index,x_center,y_center)
            if z == 0:
                z = self.getDepthFromEstimation(index,x_center,y_center)
            # print(f'depth value of object {z}')
            x = (x_center-self.ppx)/self.fx * z 
            y = (y_center-self.ppy)/self.fy * z
            xyzr.append([x,y,z,rot])
        return xyzr

    def process_hand(self, images_path):
        file = os.path.join(self.processed_path,'images',images_path,'det.csv')
        df = pd.read_csv(file, header=None)
        print(f'length of dataframe {len(df)}')
        # select the third column with bbox [x1,y1,x2,y2] 
        df[2] = df[2].apply(str2list)
        # create hand trajectory in camera frame: num_frames x [xcenter, ycenter, zcenter]
        hand_traj = self.convert_hand_bbox_to_xyz(df[2])
        return hand_traj

    def process_objects(self,images_path):
        # process all objects in the path
        for i in range(len(self.class_labels)):
            object_path = os.path.join(self.processed_path,'obb')
            obj = os.path.join(object_path,images_path) + "_" + self.class_labels[i] + ".csv"
            df = pd.read_csv(obj)
            try:
                df = df.fillna(method='ffill')
                xyzr = self.convert_obj_bbox_to_xyz(df)
            except:
                df = df.fillna(method='bfill')
                xyzr = self.convert_obj_bbox_to_xyz(df)
            create_product_positions_table(self.db_path,self.class_labels[i])
            # save data into database
            product_position_dao = ProductPositionDAO(self.db_path,self.class_labels[i])
            product_position_dao.add_product_positions(xyzr)



        