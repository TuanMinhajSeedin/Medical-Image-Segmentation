from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    copy_data: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path
