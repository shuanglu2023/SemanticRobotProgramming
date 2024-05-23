from db.database import create_tables
from language_module.language_module import LangugageModule
from env_module.env_module import EnvModule
from task_module.entityDAO import GroundedProductDAO, ProductPositionDAO
class Programming():
    def __init__(self, task_model_id, txt_path: str, video_path: str, db_path: str):
        self.task_model_id = task_model_id
        self.txt_input = txt_path
        self.video_path = video_path
        self.db_path = db_path

    def process_instruction(self):
        """
        process the instruction then save the relevant information into the database,
        call the methods defined in language module
        """
        lm = LangugageModule(self.task_model_id, self.db_path)
        predict_path = '/home/ziyu/semantic_programming/data/processed/test.json'
        lm.programming_process_text(self.txt_input,predict_path)

    def process_video(self):
        env = EnvModule(self.task_model_id, db_path)
        env.process_video(self.video_path)

    def integrate_product_position_data(self):
        # information grounding between products table and product positions table
        # the products table were created in processing instruction method
        # the product positions table were created in processing video method

        # get grounded products table
        grounded_product_dao = GroundedProductDAO(self.db_path)
        results = grounded_product_dao.get_all_products()
        for result in results:
            product_name = result[1]
            source_location_id = result[2]
            target_location_id = result[3]
            product_position_dao = ProductPositionDAO(self.db_path,product_name)

            if source_location_id is not None:
                # get grasp index, update source_location_id
                grasp_index = product_position_dao.get_grasp_index()
                product_position_dao.update_source_location(grasp_index,source_location_id)
            if target_location_id is not None:
                # get release index, update target_location_id
                release_index = product_position_dao.get_release_index()
                product_position_dao.update_target_location(release_index,target_location_id)
            # update product id
                



if __name__ == "__main__":
    db_path ="./db/tasks.db"
    create_tables(db_path)
    txt_path = "./data/raw/input.txt"
    video_path = "./data/raw/20231220_134929.bag"
    task_model_id = 1
    programming = Programming(task_model_id,txt_path,video_path,db_path)
    programming.process_instruction()
    programming.process_video()
    # programming.integrate_product_position_data()




