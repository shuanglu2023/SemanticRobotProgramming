import sqlite3
from .entity import TaskModel

class TaskModelDAO:
    def __init__(self, db_path):
        self.db_path = db_path

    def add_association(self, task_model_id, target_object_id):
        sql = "INSERT INTO TaskModelTargetObjects (TaskModelId, TargetObjectId) VALUES (?, ?);"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (task_model_id, target_object_id))
            conn.commit()

    def get_target_objects_for_task_model(self, task_model_id):
        sql = "SELECT TargetObjectId FROM TaskModelTargetObjects WHERE TaskModelId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (task_model_id,))
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def remove_association(self, task_model_id, target_object_id):
        sql = "DELETE FROM TaskModelTargetObjects WHERE TaskModelId = ? AND TargetObjectId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (task_model_id, target_object_id))
            conn.commit()

    def delete_all_entries(self):
        sql = "DELETE FROM TaskModelTargetObjects;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()

class ProductDAO:
    def __init__(self, db_path):
        self.db_path = db_path

    def add_product(self, product):
        sql = "INSERT INTO Products (ObjectId, ObjectName, TokenId, SentenceId, SourceLocationId, TargetLocationId, Color) VALUES (?, ?, ?, ?, ?, ?, ?);"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Assuming source_location_id, target_location_id, and color are optional and can be None
            # print(f'---------------insert data into db {product.object_id, product.object_name, product.token_id, product.sentence_id}')
            cursor.execute(sql, (product.object_id, product.object_name, product.token_id, product.sentence_id, getattr(product, 'source_location_id', None), getattr(product, 'target_location_id', None), getattr(product, 'color', None)))
            conn.commit()

    def get_product(self, object_id):
        sql = "SELECT * FROM Products WHERE ObjectId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (object_id,))
            return cursor.fetchone()

    def delete_all_entries(self):
        sql = "DELETE FROM Products;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()

    def check_token_id_exists(self,token_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Query to check if a record with the specified TokenId exists
            cursor.execute("SELECT EXISTS(SELECT 1 FROM Products WHERE TokenId = ?)", (token_id,))
            exists = cursor.fetchone()[0]

            return exists == 1
        
    def update_source_location(self, object_id, new_source_location_id):
        """
        Updates the source location of a product identified by its object ID.

        Parameters:
        object_id (int): The ID of the product to update.
        new_source_location_id (int): The new source location ID to set for the product.
        """
        sql = "UPDATE Products SET SourceLocationId = ? WHERE ObjectId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (new_source_location_id, object_id))
            conn.commit()

    def update_target_location(self, object_id, new_target_location_id):
        """
        Updates the source location of a product identified by its object ID.

        Parameters:
        object_id (int): The ID of the product to update.
        new_source_location_id (int): The new source location ID to set for the product.
        """
        sql = "UPDATE Products SET TargetLocationId = ? WHERE ObjectId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (new_target_location_id, object_id))
            conn.commit()

    def get_all_products(self):
        sql = "SELECT * FROM Products;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()
        
    def update_product_color(self, object_id, new_color):
        """
        Updates the color of a product identified by its object ID.

        Parameters:
        object_id (int): The ID of the product to update.
        new_color (str): The new color to set for the product.
        """
        sql = "UPDATE Products SET Color = ? WHERE ObjectId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            print(f'----------------object_id {object_id} color {new_color}')
            cursor.execute(sql, (new_color, object_id))
            conn.commit()

    def get_products_by_sentence_id(self, sentence_id):
        """
        Retrieves products from the Products table with a specific sentence_id.
        
        Parameters:
        sentence_id (int): The sentence ID to filter the products.
        """
        sql = "SELECT * FROM Products WHERE SentenceId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (sentence_id,))
            return cursor.fetchall()
        
    def get_products_by_name(self,object_name):
        sql = "SELECT * FROM Products WHERE ObjectName = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (object_name,))
            return cursor.fetchall()

class ProductPositionDAO:
    def __init__(self,db_path,product_name):
        self.db_path = db_path
        self.product_name = product_name

    def add_product_positions(self,product_positions):
        sql = f"INSERT INTO ProductPositions{self.product_name} (X, Y, Z, Rz) VALUES (?, ?, ?, ?);"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, [(pp[0],pp[1],pp[2],pp[3]) for pp in product_positions])
            conn.commit()

    def get_product_positions(self):
        sql = f"SELECT * FROM ProductPositions{self.product_name};"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()
        
    def update_grasp_index(self,time_id):
        sql = f"UPDATE ProductPositions{self.product_name} SET Grasp = ? WHERE TimeId = ?;"
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (1,time_id))
                conn.commit()
        except sqlite3.Error as e:
            print(e)
    
    def update_release_index(self,time_id):
        sql = f"UPDATE ProductPositions{self.product_name} SET Release = ? WHERE TimeId = ?;"
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (1,time_id))
                conn.commit()
        except sqlite3.Error as e:
            print(e)
    
    def update_source_location(self, grasp_index, source_location_id):
        sql = f"UPDATE ProductPositions{self.product_name} SET SourceLocationId = ? WHERE TimeId <= ?;"
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (source_location_id, grasp_index))
                conn.commit()
        except sqlite3.Error as e:
            print(e)

    def update_target_location(self, release_index, target_location_id):
        sql = f"UPDATE ProductPositions{self.product_name} SET TargetLocationId = ? WHERE TimeId >= ?;"
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (target_location_id, release_index))
                conn.commit()
        except sqlite3.Error as e:
            print(e)

    def get_grasp_index(self):
        sql = f"SELECT TimeId FROM ProductPositions{self.product_name} WHERE Grasp = 1;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is not None:
                return result[0]
            else:
                return None
            
    def get_release_index(self):
        sql = f"SELECT TimeId FROM ProductPositions{self.product_name} WHERE Release = 1;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is not None:
                return result[0]
            else:
                return None
            
    def get_reach_trajectory(self):
        # Get the trajectory from the start to the grasp index
        sql = f"SELECT X, Y, Z, Rz FROM ProductPositions{self.product_name} WHERE TimeId <= (SELECT TimeId FROM ProductPositions{self.product_name} WHERE Grasp = 1);"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()
    
    def get_move_trajectory(self):
        # Get the trajectory from the grasp index to the release index
        sql = f"SELECT X, Y, Z, Rz FROM ProductPositions{self.product_name} WHERE TimeId >= (SELECT TimeId FROM ProductPositions{self.product_name} WHERE Grasp = 1) AND TimeId <= (SELECT TimeId FROM ProductPositions{self.product_name} WHERE Release = 1);"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()

class SourceLocationDAO:
    def __init__(self, db_path):
        self.db_path = db_path

    def add_source_location(self, spatial_indicator):
        sql = "INSERT INTO SourceLocations (LocationId, Description, TrajectorObjectId) VALUES (?, ?, ?);"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql,(spatial_indicator.location_id,spatial_indicator.description,spatial_indicator.object_id))
            conn.commit()

    def get_source_location(self, location_id):
        sql = "SELECT * FROM SourceLocations WHERE LocationId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (location_id,))
            return cursor.fetchone()

    def update_source_location(self, location_id, new_location_name):
        sql = "UPDATE SourceLocations SET LocationName = ? WHERE LocationId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (new_location_name, location_id))
            conn.commit()

    def delete_source_location(self, location_id):
        sql = "DELETE FROM SourceLocations WHERE LocationId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (location_id,))
            conn.commit()

    def delete_all_entries(self):
        """
        Deletes all entries from the SourceLocations table.
        """
        sql = "DELETE FROM SourceLocations;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()

class TargetLocationDAO:
    def __init__(self, db_path):
        self.db_path = db_path

    def add_target_location(self, target_location):
        """
        Adds a new target location to the database.
        """
        sql = "INSERT INTO TargetLocations (LocationId, Description, IndicatorObjectId) VALUES (?, ?, ?);"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql,(target_location.location_id,target_location.description,target_location.object_id))
            conn.commit()

    def get_target_location(self, location_id):
        """
        Retrieves a target location by its ID.
        """
        sql = "SELECT * FROM TargetLocations WHERE LocationId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (location_id,))
            return cursor.fetchone()

    def update_target_location(self, location_id, new_description):
        """
        Updates the description of a target location.
        """
        sql = "UPDATE TargetLocations SET Description = ? WHERE LocationId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (new_description, location_id))
            conn.commit()

    def delete_target_location(self, location_id):
        """
        Deletes a target location from the database.
        """
        sql = "DELETE FROM TargetLocations WHERE LocationId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (location_id,))
            conn.commit()

    def delete_all_entries(self):
        """
        Deletes all entries from the TargetLocations table.
        """
        sql = "DELETE FROM TargetLocations;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()

class GroundedProductDAO:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_all_products(self):
        sql = "SELECT * FROM GroundedProducts;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()

    def add_grounded_product(self, class_id, product_name):
        """
        Adds a new grounded product to the database.
        """
        try:
            sql = "INSERT INTO GroundedProducts (ClassId, ObjectName) VALUES (?, ?);"
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql,(class_id,product_name))
                conn.commit()
        except sqlite3.Error as e:
            print(e)

    def update_source_location(self, class_id, source_location_id):
        """
        Updates the source location of a grounded product.
        """
        sql = "UPDATE GroundedProducts SET SourceLocationId = ? WHERE ClassId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (source_location_id, class_id))
            conn.commit()

    def update_target_location(self, class_id, target_location_id):
        """
        Updates the target location of a grounded product.
        """
        sql = "UPDATE GroundedProducts SET TargetLocationId = ? WHERE ClassId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (target_location_id, class_id))
            conn.commit()

    def update_color(self, class_id, color):
        """
        Updates the color of a grounded product.
        """
        sql = "UPDATE GroundedProducts SET Color = ? WHERE ClassId = ?;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (color, class_id))
            conn.commit()

class HandPositionDAO:
    def __init__(self,db_path,task_model_id):
        self.db_path = db_path
        self.task_model_id = task_model_id

    def add_hand_positions(self,hand_positions):
        sql = f"INSERT INTO HandPositions (X, Y, Z) VALUES (?, ?, ?);"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, [(pp[0],pp[1],pp[2]) for pp in hand_positions])
            conn.commit()

    def get_hand_positions(self):
        sql = f"SELECT * FROM HandPositions;"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()


# Example usage:
if __name__ == '__main__':
    # Relative path
    DATABASE_PATH = './db/tasks.db'
    target_object_ids = [1,2,3,4]
    task_model = TaskModel(target_object_ids)

    # Create an instance of the TaskModelDAO
    task_model_dao = TaskModelDAO(DATABASE_PATH)

    # Add the TaskModel to the database
    task_model_dao.add_task_model(task_model)