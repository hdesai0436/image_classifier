from src.image_classifers.constants import CONFIG_FILE_PATH,PARAMS_FILE_PATH
import os
from pathlib import Path
from src.image_classifers.utils.common import read_yaml ,create_directories
from src.image_classifers.entity.config_enityt import DataIngestionconfig,PreprocessConfig ,preparebasemodelconfig,PrepareCallbackConfig,TrainingConfig

class ConfigurationManager:
    def __init__(self,config_filepath = CONFIG_FILE_PATH, params_filepath= PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config['artifacts_root']])


    def get_data_ingestion_config(self)-> DataIngestionconfig:
        config = self.config['data_ingestion']
        create_directories([config['root_dir']])

        data_ingestion_config = DataIngestionconfig(
            root_dir = config['root_dir'],
            source_url = config['source_file'],
            local_data_file = config['local_data'],
            unzip_dir = config['unzip_dir']
        )
        return data_ingestion_config
    

    def data_preprocess(self)-> PreprocessConfig:
        config = self.config['preprocess_data']
        create_directories([config['root_dir']])
        train_data= os.path.join(self.config['data_ingestion']['unzip_dir'],"MURA-v1.1","train_image_paths.csv").replace("\\","/")
        
        train_config = PreprocessConfig(
            raw_data_path=Path(train_data),
            train_file_path = Path(config['train_file_path']),
            test_file_path = Path(config['test_file_path'])

        )
        return train_config
    
    def get_prepare_base_model_config(self)->preparebasemodelconfig:
        config = self.config['prepare_base_model']
        create_directories([config['root_dir']])

        prepare_base_model_config = preparebasemodelconfig(
                                    root_dir=Path(config['root_dir']),
                                    base_model_path=Path(config['base_model_path']),
                                    update_base_model_path=Path(config['updated_base_model_path']),
                                    params_image_size= self.params['IMAGE_SIZE'],
                                    params_learing_rate= self.params['LEARNING_RATE'],
                                    params_include_top=self.params['INCLUDE_TOP'],
                                    params_weoghts=self.params['WEIGHTS'],
                                    params_classes=self.params['CLASSES']

                                    )

        return prepare_base_model_config

    def get_prepare_callback_config(self)-> PrepareCallbackConfig:
        config = self.config['prepare_callbacks']
        model_ckpt_dir = os.path.dirname(config['checkpoint_model_filepath'])

        create_directories([
            Path(config['checkpoint_model_filepath']),
            Path(config['tensorboard_root_log_dir'])
        ])

        prepare_callback_config = PrepareCallbackConfig(
            root_dir=Path(config['root_dir']),
            tensorboard_root_log_dir=Path(config['tensorboard_root_log_dir']),
            checkpoint_model_filepath=Path(config['checkpoint_model_filepath'])

        
        )
        return prepare_callback_config

    def get_training_config(self) ->TrainingConfig :
        traing = self.config['training']
        prepare_base_model = self.config['prepare_base_model']
        params = self.params
        training_data = self.config['data_ingestion']
        train_csv = os.path.join(self.config['preprocess_data']['train_file_path']).replace("\\","/")
        test_csv =  os.path.join(self.config['preprocess_data']['test_file_path']).replace("\\","/")

        create_directories([
            Path(traing['root_dir'])
        ])

        training_config = TrainingConfig(
            root_dir=Path(traing['root_dir']),
            trained_model_path = Path(traing['trained_model_path']),
            updated_base_model_path=Path(prepare_base_model['updated_base_model_path']),
            training_data=training_data,
            train_csv=train_csv,
            test_csv=test_csv,
            params_epochs=params['EPOCHS'],
            params_batch_size=params['BATCH_SIZE'],
            params_is_augmentation=params['AUGMENTATION'],
            params_image_size=params['IMAGE_SIZE'])
        
        
        return training_config