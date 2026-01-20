import os
import gdown
import zipfile
from LiverTumorSegmentation import logger
from LiverTumorSegmentation.utils.common import get_size
from LiverTumorSegmentation.entity.config_entity import DataIngestionConfig
import shutil



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self) -> str:
        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            url = prefix + file_id
            gdown.download(url, zip_download_dir)
            logger.info(f"Downloaded data from {dataset_url} into {zip_download_dir}")


        except Exception as e:
            raise e

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def copy_zip_file(self):
        """
        Copy the downloaded ZIP file into the `copy_data` directory.
        Ensures the destination directory exists to avoid FileNotFoundError.
        """
        os.makedirs(self.config.copy_data, exist_ok=True)
        shutil.copy(self.config.local_data_file, self.config.copy_data)

    def copy_files(self):
        """
        Backwards-compatible wrapper used by the pipeline.
        Currently just delegates to `copy_zip_file()`.
        """
        self.copy_zip_file()

    