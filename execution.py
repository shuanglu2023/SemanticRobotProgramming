import os
from task_module.entityDAO import TaskModelDAO, ProductDAO, ProductPositionDAO, HandPositionDAO, GroundedProductDAO, TargetLocationDAO
from motion_module.motion_module import MotionModule
from language_module.language_module import LangugageModule
from env_module.env_module import EnvModule
import csv
import cv2

class Execution():
    def __init__(self, DATABASE_PATH, task_model_id) -> None:
        self.db_path = DATABASE_PATH
        self.task_model_id = task_model_id

    def run_action(self, product):
        # new_start: current end-effector position
        # grasp position: object detection
        # release position: target location, calculated based on relative position from demonstration, assuming the transformation preserved
        hand_position_dao = HandPositionDAO(self.db_path, self.task_model_id)
        hand_traj = hand_position_dao.get_hand_positions()
        self.reach(product, hand_traj)
        self.move(product, hand_traj)


    def reach(self,product_name, hand_traj):
        product_position_dao = ProductPositionDAO(self.db_path,product_name)
        motion = MotionModule(1, self.db_path)
        # get start and grasp index from product position table
        reach_indices = product_position_dao.get_reach_indicies()
        reach_indices = [index[0] for index in reach_indices]
        hand_traj = [hand_traj[i] for i in reach_indices]
        dmp = motion.represent_trajectory(hand_traj)
        new_start = hand_traj[0][2:5]
        # the object position from simulation in camera frame, when robot is at home position
        # rosrun motion_planning get_transform.py
        deltaz = 0.2
        if product_name == 'cuboid':
            pre_grasp = [5.19290215e-02,-1.16036608e-01,5.93587977e-01-deltaz]
        elif product_name == 'octagon':
            pre_grasp = [-2.29960759e-02,-1.14888179e-01,5.96634194e-01-deltaz]
        elif product_name == 'parallelogram':
            pre_grasp = [-7.81114665e-03,-1.84033430e-01,5.96209853e-01-deltaz]
        elif product_name == 'star':
            pre_grasp = [5.19312263e-02,-1.84036650e-01,5.96778074e-01-deltaz]
        Y = motion.generate_trajectory(dmp, new_start, pre_grasp)
        # save data into csv
        file_name = product_name + '_reach.csv'
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(Y)

    def move(self,product_name, hand_traj):
        product_position_dao = ProductPositionDAO(self.db_path,product_name)
        motion = MotionModule(1, self.db_path)
        # from grasp index to release index)
        move_indices = product_position_dao.get_move_indicies()
        move_indices = [index[0] for index in move_indices]
        hand_traj = [hand_traj[i] for i in move_indices]
        dmp = motion.represent_trajectory(hand_traj)
        deltaz = 0.2
        graybox = [2.20934677e-02,2.49964539e-01,5.81089232e-01]
        x = graybox[0]
        y = graybox[1]
        z = graybox[2] 
        if product_name == 'cuboid':
            post_grasp = [5.19290215e-02,-1.16036608e-01,5.93587977e-01-deltaz]
        elif product_name == 'star':
            post_grasp = [5.19312263e-02,-1.84036650e-01,5.96778074e-01-deltaz]
        elif product_name == 'parallelogram':
            post_grasp = [-7.81114665e-03,-1.84033430e-01,5.96209853e-01-deltaz]
        elif product_name == 'octagon':
            post_grasp = [-8.07017599e-03,-1.16036539e-01,5.96634194e-01-deltaz]

        # extract the spatial relation, the translation and rotation between the object and the target location can be predefined from CAD file
        if product_name == 'cuboid':
            pre_release = [x+0.005,y+0.07,z-deltaz] 
        elif product_name == 'octagon':
            pre_release = [x+0.005,y-0.045,z-deltaz] 
        elif product_name == 'parallelogram':
            pre_release = [x+0.005,y-0.015,z-deltaz]
        elif product_name == 'star':
              pre_release = [x+0.005,y+0.02,z-deltaz]
        Y = motion.generate_trajectory(dmp, post_grasp, pre_release)
        # save data into csv
        file_name = product_name + '_move.csv'
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(Y)
    

def get_task_model(id, DATABASE_PATH):
    task_model_id = 1
    task_model_dao = TaskModelDAO(DATABASE_PATH)
    product_dao = ProductDAO(DATABASE_PATH)
    results = task_model_dao.get_target_objects_for_task_model(task_model_id)
    obj_sequences = []
    for product_id in results:
        product = product_dao.get_product(product_id)
        product_name = product[1]
        if product_name not in obj_sequences:
            obj_sequences.append(product_name)
    return obj_sequences



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    DATABASE_PATH = os.path.join(script_dir,'db/tasks.db')
    task_model_id = 1
    obj_sequences = get_task_model(task_model_id, DATABASE_PATH)
    exec = Execution(DATABASE_PATH, task_model_id)

    # images captured from camera in simulation
    # im_path = "/home/ziyu/semantic_programming/data/color.jpg"
    # im = cv2.imread(im_path)
    # print(f'color iamge shape {im.shape}')
    # im_depth_path = "/home/ziyu/semantic_programming/data/depth.jpg"
    # im_depth = cv2.imread(im_depth_path)
    # print(f'depth iamge shape {im_depth.shape}')
    
    # obj_detected = EnvModule.det_obj(im)
    for product in obj_sequences:
        exec.run_action(product)
    # input_text = input("Enter the task: ")
    # # save the input into a txt file
    # file_path = 'data/raw/test.txt'
    # with open(file_path, 'w') as f:
    #     f.write(input_text)
    # # load the task from database
    # lm = LangugageModule(task_model_id, DATABASE_PATH)
    # task = lm.get_task(file_path)