import os
import zipfile
from src.image_classifer.logger import logging
from src.image_classifer.exception import image_classifier_expection
from src.image_classifer.entity.config_enityt import DataIngestionconfig
from pathlib import Path
import shutil

class DataIngestion:
    def __init__(self,config: DataIngestionconfig):
        self.config = config
       
    def get_file(self):
        source = self.config.source_url
        destionation = self.config.local_data_file
        shutil.copy(source,destionation)


    def download_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
