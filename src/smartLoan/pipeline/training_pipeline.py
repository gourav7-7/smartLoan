from smartLoan.components.data_ingestion import DataIngestion
from smartLoan.components.data_validation import DataValidation
from smartLoan.components.data_transformation import DataTransformation
from smartLoan.components.model_training import ModelTrainer
from smartLoan.components.model_evaluation import ModelEvaluator
from smartLoan.utils.logger import logger

class DataIngestionStage:
    def __init__(self):
        self.data_ingestion = DataIngestion()

    def main(self):
        logger.info(">>>>>>>>>> Data Ingestion Started <<<<<<<<<<")
        self.data_ingestion.run()
        logger.info(">>>>>>>>>> Data Ingestion Completed <<<<<<<<<<")

class DataValidationStage:
    def __init__(self):
        self.data_val = DataValidation()

    def main(self):
        logger.info(">>>>>>>>>> Data Validation Started <<<<<<<<<<")
        self.data_val.run()
        logger.info(">>>>>>>>>> Data Validation Completed <<<<<<<<<<")



class DataTransformationStage:
    def __init__(self):
        self.data_transformation = DataTransformation()

    def main(self):
        logger.info(">>>>>>>>>>>> Data Transformation started <<<<<<<<<<<<")
        self.data_transformation.transform()
        logger.info(">>>>>>>>>>>> Data Transformation completed <<<<<<<<<<<<")


class ModelTrainingStage:
    def __init__(self):
        self.model_trainer = ModelTrainer()

    def main(self):
        logger.info(">>>>>>>>>>>> Model Training started <<<<<<<<<<<<")
        self.model_trainer.train()
        logger.info(">>>>>>>>>>>> Model Training completed <<<<<<<<<<<<")

class ModelEvaluationStage:
    def __init__(self):
        self.model_eval = ModelEvaluator()

    def main(self):
        logger.info(">>>>>>>>>> Model Evaluation Started <<<<<<<<<<")
        self.model_eval.evaluate()
        logger.info(">>>>>>>>>> Model Evaluation Completed <<<<<<<<<<")


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestionStage()
        self.data_validation = DataValidationStage()
        self.transformation = DataTransformationStage()
        self.training = ModelTrainingStage()
        self.evaluating = ModelEvaluationStage()

    def run_pipeline(self):
        logger.info("========== Training Pipeline Started ==========")

        self.data_ingestion.main()
        self.data_validation.main()
        self.transformation.main()
        self.training.main()
        self.evaluating.main()

        logger.info("========== Training Pipeline Completed ==========")


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.exception(e)
        raise e