from src.image_classifers.constants import CONFIG_FILE_PATH,PARAMS_FILE_PATH
import os
from pathlib import Path
from src.image_classifers.utils.common import read_yaml ,create_directories
from src.image_classifers.entity.config_enityt import DataIngestionconfig,PreprocessConfig ,preparebasemodelconfig

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
