import os
from task_module.entityDAO import TaskModelDAO, ProductDAO, ProductPositionDAO, HandPositionDAO, GroundedProductDAO, TargetLocationDAO
from motion_module.motion_module import MotionModule
from language_module.language_module import LangugageModule
import csv


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
            pre_grasp = [-8.07017599e-03,-1.16036539e-01,5.96634194e-01-deltaz]
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

        grounded_product_dao = GroundedProductDAO(self.db_path)
        target_location_dao = TargetLocationDAO(self.db_path)
        product_dao = ProductDAO(self.db_path)
        # extract the spatial relation, the translation and rotation between the object and the target location can be predefined from CAD file
        if product_name == 'cuboid':
            # get the target location
            # target_location_id = grounded_product_dao.get_target_location_by_product_name(product_name)
            # print(f'target location id: {target_location_id[0]}')
            # result = target_location_dao.get_target_location(target_location_id[0])
            # print(f'target location: {result}')
            # product_id = int(result[1])
            # product = product_dao.get_product(product_id)
            # print(f'product name: {product}')
            # product_position_dao = ProductPositionDAO(self.db_path,product[4]+product[1])
            # position = product_position_dao.get_product_positions_by_time_id(move_indices[-1]+1)
            # print(f'graybox position: {position[5], position[6], position[7], position[10]}')
            # x = position[5]
            # y = position[6]
            # z = position[7]
            # product_position_dao = ProductPositionDAO(self.db_path,product_name)
            # position = product_position_dao.get_product_positions_by_time_id(move_indices[-1]+1)
            # print(f'{product_name} position: {position[5], position[6], position[7], position[10]}')
            # print(f'position: {x-position[5], y-position[6], z-position[7]}')
            # pre_release = [0.021+0.06,0.2499,0.581-0.1]
            # pre_release = [x+0.08,y,z-deltaz]
            pre_release = [x,y+0.07,z-deltaz] 
        elif product_name == 'octagon':
            # pre_release = [0.021-0.03,0.2499,0.581-0.1]
            # pre_release = [x-0.055,y,z-deltaz] 
            pre_release = [x,y-0.045,z-deltaz] 
        elif product_name == 'parallelogram':
            # pre_release = [0.021,0.2499,0.581-0.1]
            # pre_release = [x-0.029,y,z-deltaz]
            pre_release = [x,y-0.01,z-deltaz]
        elif product_name == 'star':
            # pre_release = [0.021+0.03,0.2499,0.581-0.1]
            # pre_release = [x+0.035,y,z-deltaz]
              pre_release = [x,y+0.02,z-deltaz]
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
    # obj_sequences = get_task_model(task_model_id, DATABASE_PATH)
    # exec = Execution(DATABASE_PATH, task_model_id)

    # for product in obj_sequences:
    #     exec.run_action(product)
    input_text = input("Enter the task: ")
    # load the task from database
    lm = LangugageModule(task_model_id, DATABASE_PATH)
    task = lm.get_task(input_text)