import os
from task_module.entityDAO import TaskModelDAO, ProductDAO, ProductPositionDAO, HandPositionDAO
from motion_module.motion_module import MotionModule
import csv

class Execution():
    def __init__(self, DATABASE_PATH, task_model_id) -> None:
        self.db_path = DATABASE_PATH
        self.task_model_id = task_model_id

    def run_action(self, product,initial_index):
        # new_start: current end-effector position
        # grasp position: object detection
        # release position: target location, calculated based on relative position from demonstration, assuming the transformation preserved
        hand_position_dao = HandPositionDAO(self.db_path, self.task_model_id)
        hand_traj = hand_position_dao.get_hand_positions()
        self.reach(product, hand_traj, initial_index)
        release_index = self.move(product, hand_traj)
        return release_index

    def reach(self,product_name, hand_traj, initial_index):
        product_position_dao = ProductPositionDAO(self.db_path,product_name)
        motion = MotionModule(1, self.db_path)
        # from start to grasp index
        grasp_index = product_position_dao.get_grasp_index()
        hand_traj = hand_traj[initial_index:grasp_index]
        dmp = motion.represent_trajectory(hand_traj)
        new_start = hand_traj[0][2:5]
        if product_name == 'cuboid':
            pre_grasp = [0.055126707,-0.01569,0.5968-0.1]
        elif product_name == 'octagon':
            pre_grasp = [-0.0115568,-0.11628529,0.5968-0.1]
        elif product_name == 'parallelogram':
            pre_grasp = [0.0080519,-0.163,0.5968-0.1]
        elif product_name == 'star':
            pre_grasp = [0.0557,-0.1840297,0.5968-0.1]
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
        grasp_index = product_position_dao.get_grasp_index()
        release_index = product_position_dao.get_release_index()
        print(f'release index {release_index}')
        hand_traj = hand_traj[grasp_index:release_index]
        dmp = motion.represent_trajectory(hand_traj)
        if product_name == 'cuboid':
            post_grasp = [0.055126707,-0.01569,0.5968-0.1]
        elif product_name == 'octagon':
            post_grasp = [-0.0115568,-0.11628529,0.5968-0.1]
        elif product_name == 'parallelogram':
            post_grasp = [0.0080519,-0.163,0.5968-0.1]
        elif product_name == 'star':
            post_grasp = [0.0557,-0.1840297,0.5968-0.1]

        if product_name == 'cuboid':
            pre_release = [0.021+0.06,0.2499,0.581-0.1]
        elif product_name == 'octagon':
            pre_release = [0.021-0.03,0.2499,0.581-0.1]
        elif product_name == 'parallelogram':
            pre_release = [0.021,0.2499,0.581-0.1]
        elif product_name == 'star':
            pre_release = [0.021+0.03,0.2499,0.581-0.1]

        Y = motion.generate_trajectory(dmp, post_grasp, pre_release)
        # save data into csv
        file_name = product_name + '_move.csv'
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(Y)
        return release_index
    

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
    initial_index = 96
    for product in obj_sequences:
        print(f'initial index: {initial_index}')
        final_index = exec.run_action(product, initial_index)
        initial_index = final_index
    # input_text = input("Enter the task: ")
    # load the task from database
    # task = get_task(input_text)