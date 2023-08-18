import os
from src.image_classifers.logger import logging
from src.image_classifers.exception import image_classifier_expection
from pathlib import Path
import yaml
def read_yaml(path_to_yam: Path):
    logging.info('read the yeam file')
    try:
        with open(path_to_yam) as yaml_file:
            content = yaml.safe_load(yaml_file)
        return content
    except Exception as e:
        raise e


def create_directories(path_to_directories:list):
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)

        
