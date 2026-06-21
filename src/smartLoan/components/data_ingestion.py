import os
import zipfile
import subprocess
from pathlib import Path

from smartLoan.utils.logger import logger
from smartLoan.config import paths


class DataIngestion:
    def __init__(self):
        self.source = "uciml/default-of-credit-card-clients-dataset"
        self.download_dir = paths.ARTIFACTS_DIR
        self.raw_data_dir = paths.RAW_DATA_DIR
        self.target_file = "UCI_Credit_Card.csv"

    def download_file(self):
        try:
            os.makedirs(self.download_dir, exist_ok=True)

            logger.info(f"Downloading Kaggle dataset: {self.source}")

            subprocess.run(
                [
                    "kaggle",
                    "datasets",        
                    "download",
                    "-d",              
                    self.source,      
                    "-p",
                    str(self.download_dir),
                    "--force"
                ],
                check=True
            )

            logger.info(f"Download completed at {self.download_dir}")

        except Exception as e:
            logger.error("Error during Kaggle download")
            raise e

    def extract_zip_file(self):
        try:
            os.makedirs(self.raw_data_dir, exist_ok=True)

            zip_files = [
                f for f in os.listdir(self.download_dir)
                if f.endswith(".zip")
            ]

            if not zip_files:
                raise FileNotFoundError(
                    f"No zip file found in {self.download_dir}"
                )

            zip_path = Path(self.download_dir) / zip_files[0]

            logger.info(f"Extracting {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_dir)

            # Keep only the target CSV, remove everything else
            for f in os.listdir(self.raw_data_dir):
                if f != self.target_file:
                    os.remove(os.path.join(self.raw_data_dir, f))

            target_path = Path(self.raw_data_dir) / self.target_file
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Expected file '{self.target_file}' not found after extraction. "
                    f"Files present: {os.listdir(self.raw_data_dir)}"
                )

            logger.info(f"Extraction completed at {self.raw_data_dir}")

        except Exception as e:
            logger.error("Error during extraction")
            raise e

    def run(self):
        self.download_file()
        self.extract_zip_file()