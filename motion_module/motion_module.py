# movement_primitives
# read data from database
# represent the trajectory as DMP
from typing import Any
from movement_primitives.dmp import DMP
import numpy as np
from motion_module.cam_utils import get_pose

class MotionModule():
    def __init__(self, task_model_id: int, db_path: str):
        self.task_model_id = task_model_id
        self.db_path = db_path

    def represent_trajectory(self, data: Any) -> Any:
        """
        represent the trajectory as DMP
        data: list of tuples, each tuple contains the hand positions (TimeId, TaskModelId, X,Y,Z)
        """
        traj = []
        for row in data:
            traj.append(row[2:5])
        # transform trajectroy in camera frame to robot base frame   
        traj = np.array(traj)
        dmp = DMP(n_dims=3, dt=0.05, n_weights_per_dim=20)
        T = np.linspace(0.0, 1.0, len(traj))
        Y = np.empty((len(T), 3))
        Y[:,0] = traj[:,0]
        Y[:,1] = traj[:,1]
        Y[:,2] = traj[:,2]
        dmp.imitate(T, Y, regularization_coefficient=0.1)
        return dmp

    def generate_trajectory(self, dmp: Any, new_start, new_goal) -> None:
        """
        save the dmp into database
        """
        new_start = np.array(new_start)
        new_goal = np.array(new_goal)
        dmp.configure(start_y= new_start, goal_y=new_goal)
        T,Y = dmp.open_loop()
        return Y

# test usage
if __name__ == '__main__':
    db_path = './db/tasks.db'
    task_model_id = 1
    motion_module = MotionModule(task_model_id, db_path)
    data = motion_module.read_data('trajectory')
    dmp = motion_module.represent_trajectory(data)
    motion_module.save_dmp(dmp)