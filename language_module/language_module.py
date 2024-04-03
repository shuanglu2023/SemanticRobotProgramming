"""
This is a module to process the text input
"""
import subprocess
import os
from .coreference_resolution import corref
import spacy
from spacy import displacy
import json
from spacy.lang.en import English
import configparser
from collections import defaultdict
from task_module.entity import Action, Product, TaskModelTargetObject, SourceLocation, TargetLocation
from task_module.entityDAO import TaskModelDAO, ProductDAO, SourceLocationDAO, TargetLocationDAO

# helper functions
def read_config_file(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def write_config_file(config, file_path):
    with open(file_path, 'w') as configfile:
        config.write(configfile)

def token2sent(tokens):
    sent = ""
    stop_list = [',','.',"'"]
    for t in tokens:
        if t in stop_list:
            sent = sent.rstrip() + t + " "
        else:
            sent = sent + t + " "
    return sent.rstrip()


def vis(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    displacy.serve(doc, style="dep")



class LangugageModule:
    """Class to handle the processing of text"""
    def __init__(self,task_model_id,db_path):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Define the absolute path of spert
        self.spert_path = os.path.join(script_dir,'spert')
        self.model_location = os.path.join(self.spert_path,'data/save/train/final_model')
        # Define the processed data path, parent directory of the current script
        self.parent_directory = os.path.dirname(script_dir)
        self.processed_path = os.path.join(self.parent_directory,'data/processed')
        self.prediction_path = os.path.join(self.processed_path,"predictions.json")


    def train_classifier(self):
        """train spert classifier for NER and Relation Extraction"""
        # Change directory to spert
        os.chdir(self.spert_path)
        # Define the command and arguments
        command = ["python", "./spert.py", "train", "--config", "configs/train.conf"]
        # Run the command
        result = subprocess.call(command)

    def run_classifier(self,file_path):
        """
        predict using the trained classifier
        make sure the config file is correctly defined
        save predictions results in file
        """
        # Change directory to spert
        os.chdir(self.spert_path)
        config = read_config_file("configs/predict.conf")
        config['1']['dataset_path'] = file_path
        config['1']['predictions_path'] = self.prediction_path
        write_config_file(config,"configs/predict.conf")
        # Define the command and arguments
        command = ["python", "./spert.py", "predict", "--config", "configs/predict.conf"]
        # Run the command
        result = subprocess.call(command)

    def dep_parser(self):
        nlp = spacy.load("en_core_web_sm")
        # read predictions.json into memory and process it
        with open(self.prediction_path) as f:
            predict_json = json.load(f)
        
        # save the results into database
        DATABASE_PATH = os.path.join(self.parent_directory,'db/tasks.db')
        task_model_id = 1
        task_model_dao = TaskModelDAO(DATABASE_PATH)
        task_model_dao.delete_all_entries()

        product_dao = ProductDAO(DATABASE_PATH)
        product_dao.delete_all_entries()

        source_location_dao = SourceLocationDAO(DATABASE_PATH)
        source_location_dao.delete_all_entries()

        target_location_dao = TargetLocationDAO(DATABASE_PATH)
        target_location_dao.delete_all_entries()

        source_location_id = 0
        target_location_id = 0
        product_id = 0

        # for each sentence
        for i in range(len(predict_json)):
            sentence_id = i
            x = predict_json[i]
            tokens = x['tokens']
            entities = x['entities']
            relations = x['relations']

            actions = []
            colors = []
            spatial_relations = defaultdict(list)
            move_relations = defaultdict(set)
            # find token span of Action
            for entity in entities:
                action = {}
                if entity["type"] == "Action":
                    action['start'] = entity['start']
                    action['end'] = entity['end']
                    # assume each action has only one token, save ids of token
                    actions.append(action['start'])
                elif entity["type"] == "Color":
                    start = entity['start']
                    end = entity['end']
                    # print(f'{start} {end} {tokens[start:end]}')
                    colors.append(tokens[start:end])

            try:
                for relation in relations:
                    if relation['type']=='Spatial':
                        trajector = entities[relation['head']]
                        spatial_indicator = entities[relation['tail']]
                        spatial_relations[trajector['start']].append([spatial_indicator['start'],spatial_indicator['end']])
                        
                    elif relation['type']=='Transition':
                        action = entities[relation['head']]
                        goal_indicator = entities[relation['tail']]
                        # print(action, goal_indicator)
                        move_relations[action['start']].add(goal_indicator['start'])
                        move_relations[action['start']].add(goal_indicator['end'])
            except:
                pass

            sentence = token2sent(tokens)
            print('------------------------------------------------------------------------')
            print(f'{sentence}')
            print('------------------------------------------------------------------------')

            doc = nlp(sentence)
            for i in range(len(doc)):
                token = doc[i]
                # i is an action, find its direct obj
                if(token.i in actions):
                    # print(f'action {token.i}')
                    for child in token.children:
                        if child.dep_ == 'dobj':
                            target_object = Product(product_id, child.text, child.i, sentence_id)
                            product_dao.add_product(target_object)
                            action = Action(token.i,token.lemma_,product_id)
                            task_model_dao.add_association(task_model_id,product_id)
                            product_id = product_id + 1
                            # check if the object is trajector and if the object has source location
                            if child.i in spatial_relations:
                                span_lists = list(spatial_relations[child.i])
                                for span_list in span_lists:
                                    start = span_list[0]
                                    end = span_list[1]
                                    if start > end:
                                        start, end = end, start
                                    for child in doc[end-1].children:
                                        if child.dep_ == 'pobj':
                                            result = product_dao.check_token_id_exists(child.i)
                                            if not result:
                                                spatial_object = Product(product_id,child.text,child.i,sentence_id)
                                                product_dao.add_product(spatial_object)
                                                spatial_indicator = SourceLocation(source_location_id,doc[start:end].text,product_id)

                                                source_location_dao.add_source_location(spatial_indicator)
                                                target_object.set_source_location(source_location_id)
                                                product_dao.update_source_location(target_object.object_id, source_location_id)
                                                source_location_id += 1
                                                product_id+=1

                            # check if there are move relations
                            # print(f'token.i {token.i}')
                            if(token.i in move_relations):
                                span_list = list(move_relations[token.i])
                                start = span_list[0]
                                end = span_list[1]
                                if start > end:
                                    start, end = end, start
                                # goal_indicator = GoalIndicator(start,end, doc[start:end].text)
                                for child in doc[end-1].children:
                                    if child.dep_ == 'pobj':
                                        result = product_dao.check_token_id_exists(child.i)
                                        if not result:
                                            goal_object = Product(product_id,child.text,child.i,sentence_id)
                                            product_dao.add_product(goal_object)
                                            target_location = TargetLocation(target_location_id,doc[start:end].text,product_id)
                                            target_location_dao.add_target_location(target_location)
                                            target_object.set_target_location(target_location_id)
                                            product_dao.update_target_location(target_object.object_id, target_location_id)
                                            target_location_id += 1
                                            product_id+=1

            results = product_dao.get_products_by_sentence_id(sentence_id)

            for result in results:
                product = Product(result[0], result[1], result[-2], result[-1])
                token = doc[product.token_id]
                for child in token.children:
                    for color in colors:
                        if child.text in color:   
                            product_dao.update_product_color(product.object_id,child.text)



    def get_corref(self,input):
        return corref(input)

    def process_text(self,data_path):
        file_path = self.read_txt_from_path(data_path)
        self.run_classifier(file_path)
        self.dep_parser()
    

    def read_txt_from_path(self,data_path):
        """
        read the input txt
        perform paragraph to sentence
        perform correference resolution
        perform sentence to tokens
        save the tokens in json
        """
        nlp = English()
        nlp.add_pipe('sentencizer')
        with open(data_path,'r') as file:
            file_contents = file.readlines()
            output = []
            for line in file_contents:
                text = line.strip()
                corref_text = self.get_corref(text)
                doc = nlp(corref_text)
                sentences = [sent.text.strip() for sent in doc.sents]
                for s in sentences:
                    doc = nlp(s)
                    tokens = []
                    for token in doc:
                        tokens.append(token.text)
                    output.append({"tokens":tokens})
                
            json_object = json.dumps(output)
            file_name = "input.json"
            with open(os.path.join(self.processed_path,file_name),"w") as outfile:
                outfile.write(json_object)
        return os.path.join(self.processed_path,file_name)




# test usage
if __name__ == '__main__':
    sentence = "Then, pick up the star and position the star carefully next to the cuboid in the gray box."
    vis(sentence)