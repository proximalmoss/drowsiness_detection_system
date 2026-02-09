import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        if file_path.endswith('.h5'):
            obj.save(file_path)
        else:
            with open(file_path,"wb") as file_obj:
                dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report={}

        for i in range(len(list(models))):
            model_func=list(models.values())[i]
            model_name=list(models.keys())[i]
            para=params[model_name]

            model=model_func()

            model.fit(x_train, y_train, validation_split=0.2, epochs=para['epochs'], batch_size=para['batch_size'], verbose=0)

            y_test_pred=model.predict(x_test, verbose=0)

            y_test_pred_classes=np.argmax(y_test_pred, axis=1)
            y_test_classes=np.argmax(y_test, axis=1)

            test_model_score=accuracy_score(y_test_classes, y_test_pred_classes)

            report[model_name]=test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)