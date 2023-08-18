from src.image_classifers.config.configuration import ConfigurationManager
from src.image_classifers.components.prepae_base_model import PrepareBaseModel

class PrepareBaseModelTraningPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

if __name__ == '__main__':
    obj = PrepareBaseModelTraningPipeline()
    obj.main()