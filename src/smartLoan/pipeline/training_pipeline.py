from smartLoan.components.data_ingestion import DataIngestion
from smartLoan.utils.logger import logger

class DataIngestionStage:
    def __init__(self):
        pass

    def main(self):
        data_ingestion = DataIngestion()
        data_ingestion.run()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>>>>>> Data Ingestion started <<<<<<<<<<<<<<<")
        obj = DataIngestionStage()
        obj.main()
        logger.info(f">>>>>>>>>>>>>>> Data Ingestion Completed <<<<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
        