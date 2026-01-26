import os
import sys
from src.exception import CustomException
from src.logger import logging

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifact", "train.npy")
    test_data_path: str=os.path.join("artifact", "test.npy")
    train_labels_path: str=os.path.join("artifact", "train_labels.npy")
    test_labels_path: str=os.path.join("artifact", "test_labels.npy")
    raw_data_path: str=os.path.join("artifact", "raw_data.npy")
    raw_labels_path: str=os.path.join("artifact", "raw_labels.npy")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")

        try:
            img_size=128
            data_path='notebook/data'
            classes=['Closed', 'no_yawn', 'Open', 'yawn']

            logging.info("Loading image dataset from data folder")

            images=[]
            labels=[]

            for label, cls in enumerate(classes):
                cls_path=os.path.join(data_path, cls)
                logging.info("Loading images from class")

                for img_name in os.listdir(cls_path):
                    if img_name.endswith(('.jpg','.jpeg','.png')):
                        img_path=os.path.join(cls_path, img_name)
                        img=cv2.imread(img_path)
                        if img is not None:
                            img=cv2.resize(img,(img_size, img_size))
                            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            labels.append(label)
            x=np.array(images)
            y=np.array(labels)

            logging.info("Dataset loaded")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            np.save(self.ingestion_config.raw_data_path, x)
            np.save(self.ingestion_config.raw_labels_path, y)
            logging.info("Copy of dataset saved")

            logging.info("Train test split initiated")
            x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

            np.save(self.ingestion_config.train_data_path, x_train)
            np.save(self.ingestion_config.test_data_path, x_test)
            np.save(self.ingestion_config.train_labels_path, y_train)
            np.save(self.ingestion_config.test_labels_path, y_test)

            logging.info("Data ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_labels_path,
                self.ingestion_config.test_labels_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data, train_labels, test_labels=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data, train_labels, test_labels)