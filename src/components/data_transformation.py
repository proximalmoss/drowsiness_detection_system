import os
import sys
from dataclasses import dataclass

import numpy as np
from tensorflow.keras.utils import to_categorical

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifact", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            preprocessor_config={
                'normalization':True,
                'normalize_factor':255.0,
                'img_size':128,
                'num_classes':4,
                'classes':['Closed', 'yawm', 'Open', 'yawn']
            }

            logging.info("Image preprocessing configuration created")
            logging.info('Normalization: pixel values will be scaled to 0-1 range')

            return preprocessor_config
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path, train_labels_path, test_labels_path):
        try:
            x_train=np.load(train_path)
            x_test=np.load(test_path)
            y_train=np.load(train_labels_path)
            y_test=np.load(test_labels_path)

            logging.info("Reading train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            if preprocessing_obj['normalization']:
                x_train=x_train/preprocessing_obj['normalize_factor']
                x_test=x_test/preprocessing_obj['normalize_factor']
            logging.info("Image normalization compeleted")

            num_classes=preprocessing_obj['num_classes']
            y_train=to_categorical(y_train, num_classes=num_classes)
            y_test=to_categorical(y_test, num_classes=num_classes)
            logging.info("Label encoding completed")

            train_arr=(x_train, y_train)
            test_arr=(x_test, y_test)

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)