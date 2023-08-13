from src.image_classifers.entity.config_enityt import PreprocessConfig
from src.image_classifers.config.configuration import ConfigurationManager
import pandas as pd
from sklearn.model_selection import train_test_split

class propress:
    def __init__(self, config: PreprocessConfig):
        self.config = config
    
    def preprocess_data(self):
        df = pd.read_csv(self.config.raw_data_path,header=None)
        df.columns = ['image_path']
        df['label'] = df['image_path'].apply((lambda x:'positive' if 'positive' in x else 'negative'))
        df['category'] = df['image_path'].apply(lambda x: x.split('/')[2])                                     
        df['patient_id'] = df['image_path'].apply(lambda x: x.split('/')[3].replace('patient',''))
        train_image,val_image = train_test_split(df,test_size=0.2, random_state=45)
        train_image.to_csv(self.config.train_file_path, index=False)
        val_image.to_csv(self.config.test_file_path, index=False)

