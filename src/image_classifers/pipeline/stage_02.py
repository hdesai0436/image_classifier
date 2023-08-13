from src.image_classifers.config.configuration import ConfigurationManager
from src.image_classifers.components.preprocess import propress

class DataPreprocessPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocess_config = config.data_preprocess()
        data_preprocess = propress(config=data_preprocess_config)
        data_preprocess.preprocess_data()

if __name__ == '__main__':
    obj = DataPreprocessPipeline()
    obj.main()