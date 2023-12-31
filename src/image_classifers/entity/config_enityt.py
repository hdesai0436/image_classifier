from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionconfig:
    root_dir:Path
    source_url: str
    local_data_file:Path
    unzip_dir: Path

@dataclass(frozen=True)
class PreprocessConfig:
    raw_data_path: Path
    train_file_path: Path
    test_file_path: Path


@dataclass(frozen=True)
class preparebasemodelconfig:
    root_dir: Path
    base_model_path: Path
    update_base_model_path : Path
    params_image_size: list
    params_learing_rate:float
    params_include_top: bool
    params_weoghts: str
    params_classes: int

@dataclass(frozen=True)
class PrepareCallbackConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    train_csv: Path
    test_csv: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    