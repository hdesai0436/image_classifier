from src.image_classifer.config.configuration import ConfigurationManager
from src.image_classifer.components.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestin_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestin_config)
        data_ingestion.get_file()
        data_ingestion.download_file()


if __name__ == '__main__':
    obj = DataIngestionPipeline()
    obj.main()