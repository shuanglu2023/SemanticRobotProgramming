"""
This is a module for task representation.
It handles the interface with database.
"""
from typing import List, Optional
class Action:
    def __init__(self, action_id: int, action_name: str, target_object_id: int):
        self.action_id = action_id
        self.action_name = action_name
        self.target_object_id = target_object_id

class Product:
    """
    object_id is single identifier
    object_name can be the same
    """
    def __init__(self, object_id: int, object_name: str, token_id:int, sentence_id:int):
        self.object_id = object_id
        self.object_name = object_name
        self.token_id = token_id
        self.sentence_id = sentence_id
        self.source_location_id = None  # Initialize with default value
        self.target_location_id = None  # Initialize with default value
        self.color = None  # Initialize with default value


    def set_source_location(self,source_location_id):
        self.source_location_id = source_location_id

    def set_target_location(self,target_location_id):
        self.target_location_id = target_location_id

    def set_color(self,color):
        self.color = color



class SourceLocation:
    def __init__(self, location_id: int, description: str, object_id: int):
        self.location_id = location_id
        self.description = description
        self.object_id = object_id


class TargetLocation:
    def __init__(self, location_id: int, description: str, object_id: int):
        self.location_id = location_id
        self.description = description
        self.object_id = object_id

class TaskModel:
    def __init__(self):
        self.target_object_ids = []
        self.static_object_ids = []
    def add_target_objects(self,target_object_id):
        self.target_object_ids.append(target_object_id)
    def add_static_objects(self,object_id):
        self.static_object_ids.append(object_id)

class TaskModelTargetObject:
    def __init__(self, task_model_id: int, target_object_id: int):
        self.task_model_id = task_model_id
        self.target_object_id = target_object_id
