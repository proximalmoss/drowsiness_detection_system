import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path='artifact/model.h5'
            preprocessor_path='artifact/preprocessor.pkl'

            model=load_model(model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            img_size=preprocessor['img_size']

            if preprocessor['normalization'] and features.max()>1:
                features=features/preprocessor['normalize_factor']

            preds=model.predict(features, verbose=0)
            pred_class=np.argmax(preds, axis=1)

            class_names=preprocessor['classes']
            pred_label=class_names[pred_class[0]]

            return pred_label, preds[0]
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, image_path: str):
        self.image_path=image_path
    
    def get_data_as_array(self):
        try:
            preprocessor_path='artifact/preprocessor.pkl'
            preprocessor=load_object(file_path=preprocessor_path)
            img_size=preprocessor['img_size']

            img=cv2.imread(self.image_path)
            if img is None:
                raise ValueError(f"Could not read image from {self.image_path}")
            
            img=cv2.resize(img, (img_size, img_size))
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=img/255.0

            img=np.expand_dims(img, axis=0)

            return img
        except Exception as e:
            raise CustomException(e,sys)