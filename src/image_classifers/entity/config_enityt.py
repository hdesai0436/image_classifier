from pathlib import Path
from dataclasses import dataclass

@dataclass()
class DataIngestionconfig:
    root_dir:Path
    source_url: str
    local_data_file:Path
    unzip_dir: Path

@dataclass()
class PreprocessConfig:
    raw_data_path: Path
    train_file_path: Path
    test_file_path: Path
@dataclass
class preparebasemodelconfig:
    root_dir: Path
    base_model_path: Path
    update_base_model_path : Path
    params_image_size: list
    params_learing_rate:float
    params_include_top: bool
    params_weoghts: str
    params_classes: int