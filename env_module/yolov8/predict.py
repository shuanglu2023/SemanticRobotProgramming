from ultralytics import YOLO
import os
import pandas as pd
from db.database import create_product_positions_table

def output_obb(path,images_path,width,height,class_labels):
    """Detect objects on each image, save the results in csv file"""
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"runs/obb/train/weights/best.pt")
    output_path = "/home/ziyu/semantic_programming"
    # running the model on realworld images
    model = YOLO(model_path)
    results = model.predict(source=path, stream=False, save=False, save_txt=False) # Display preds. Accepts all YOLO predict arguments
    label2id = {label: id for id, label in enumerate(class_labels)}
    id2label = {id: label for label, id in label2id.items()}
    num_dataframes = len(class_labels)
    # create number of dataframes = number of objects of interests
    dataframes = {}
    dataframes_name = []
    for i in range(num_dataframes):
        dataframes[f'df{i}'] = pd.DataFrame(columns=['Path','TimeId','Confidence','X','Y','W','H','R'])
        dataframes_name.append(f'df{i}')

    ix = 0
    # considering only one object from each class
    for result in results:
        for dfn in dataframes_name:
            df = dataframes[dfn]
            df.loc[len(df)] = None
            df.loc[ix,'Path'] = result.path
        bbox = result.obb.xywhr.to("cpu").numpy()
        confidence = result.obb.conf.to("cpu").numpy()
        cls = result.obb.cls.to("cpu").numpy()
        for i in range(len(cls)):
            idx = int(cls[i])
            dfn = dataframes_name[idx]
            df = dataframes[dfn]
            df.loc[ix,'Confidence'] = confidence[i]
            df.loc[ix,'X'] = bbox[i][0]/width
            df.loc[ix,'Y'] = bbox[i][1]/height
            df.loc[ix,'W'] = bbox[i][2]/width
            df.loc[ix,'H'] = bbox[i][3]/height
            df.loc[ix,'R'] = bbox[i][4]
            df.loc[ix,'TimeId'] = ix
        ix = ix + 1

    for i in range(len(dataframes_name)):
        file_name = os.path.join(output_path,'data','processed','obb',images_path) + '_' + id2label[i] + '.csv'
        dfn = dataframes_name[i]
        df = dataframes[dfn]
        df.to_csv(file_name,index=False, date_format=float)


def det_im(im):
    # detect objects return dict = {class: [x,y,w,h,r]}
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"runs/obb/train/weights/best.pt")
    # running the model on realworld images
    model = YOLO(model_path)
    results = model.predict(source=im, stream=False, save=True, save_txt=False) # Display preds. Accepts all YOLO predict arguments
    return results