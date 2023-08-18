import os
import tensorflow as tf
from src.image_classifers.entity.config_enityt import PrepareCallbackConfig

class PrepareCallback:
    def __init__(self,config: PrepareCallbackConfig):
        self.config = config

    @property
    def _create_ckpt_callback(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_model_filepath, 'fe.h5'),
            save_best_only=True
        )
    
    def get_tb_ckpt_callback(self):
        return [self._create_ckpt_callback]
