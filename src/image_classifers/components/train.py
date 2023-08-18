from src.image_classifers.entity.config_enityt import TrainingConfig
import tensorflow as tf
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

class Traning:
    def __init__(self,config:TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.saving.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        idg_train_settings = dict(samplewise_center = True,
                         samplewise_std_normalization = True,
                          rotation_range = 5,
                          width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         horizontal_flip = False,
                         vertical_flip = True)
        
        df = pd.read_csv(self.config.train_csv)
        test_df = pd.read_csv(self.config.test_csv)
        print(self.config.training_data['root_dir'])
        if self.config.params_is_augmentation:
            train_datagen = ImageDataGenerator(**idg_train_settings)
           
        else: 
            train_datagen = ImageDataGenerator(rescale=1./255)

        self.train_datageneraor = train_datagen.flow_from_dataframe(
                dataframe=df,
                directory=self.config.training_data['root_dir'],
                x_col='image_path',
                y_col='label',
                target_size=(224,224),
                batch_size=32,
                class_mode='binary'
            )
        
        valid_datagen = ImageDataGenerator(rescale=1./255)

        self.validation_datagen = valid_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=self.config.training_data['root_dir'],
            x_col = 'image_path',
            y_col = 'label',
            target_size = (224,224),
            batch_size=32,
             class_mode='binary'


        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.model.fit(
            self.train_datageneraor,
            epochs = 1,
            validation_data=self.validation_datagen,
            callbacks = callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model = self.model
        )


        