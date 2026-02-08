import os
import sys
from dataclasses import dataclass

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact", "model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")
            x_train, y_train, x_test, y_test=(
                train_array[0],
                train_array[1],
                test_array[0],
                test_array[1]
            )

            image_size=128
            num_classes=4

            logging.info(f"Training data shape: {x_train.shape}, Test data shape: {x_test.shape}")

            def simple_cnn():
                model=Sequential([
                    Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 3)),
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation='relu'),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(num_classes, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                return model
            def deep_cnn():
                model=Sequential([
                    Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 3)),
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation='relu'),
                    MaxPooling2D(2,2),
                    Conv2D(128, (3,3), activation='relu'),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(num_classes, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                return model
            def batchnorm_cnn():
                model=Sequential([
                    Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 3)),
                    BatchNormalization(),
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation='relu'),
                    BatchNormalization(),
                    MaxPooling2D(2,2),
                    Conv2D(128, (3,3), activation='relu'),
                    BatchNormalization(),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(num_classes, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                return model
            
            models={"Simple CNN": simple_cnn, "Deep CNN": deep_cnn, "BatchNorm CNN": batchnorm_cnn}
            params={
                "Simple CNN":{'epochs':15, 'batch_size':32},
                "Deep CNN":{'epochs':15, 'batch_size':32},
                "BatchNorm CNN":{'epochs':15, 'batch_size':32}
            }

            logging.info("Evaluating models")
            model_report:dict=evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model_func=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found: {best_model_name} with accuracy: {best_model_score}")

            logging.info("Training best model with more epochs")
            best_model=best_model_func()

            best_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model saved successfully")

            predicted=best_model.predict(x_test, verbose=0)
            y_pred_classes=np.argmax(predicted, axis=1)
            y_test_classes=np.argmax(y_test, axis=1)

            accuracy=accuracy_score(y_test_classes, y_pred_classes)

            return accuracy
        except Exception as e:
            raise CustomException(e,sys)