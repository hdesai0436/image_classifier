from src.image_classifers.config.configuration import ConfigurationManager
from src.image_classifers.components.prepare_callback import PrepareCallback
from src.image_classifers.components.train import Traning

class ModelTraingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_call_config = config.get_prepare_callback_config()
        prepare_callback = PrepareCallback(config=prepare_call_config)
        callback_list = prepare_callback.get_tb_ckpt_callback()

        train_config = config.get_training_config()
        traning = Traning(config=train_config)
        traning.get_base_model()
        traning.train_valid_generator()
        traning.train(callback_list=callback_list)


if __name__ == '__main__':
    obj = ModelTraingPipeline()
    obj.main()