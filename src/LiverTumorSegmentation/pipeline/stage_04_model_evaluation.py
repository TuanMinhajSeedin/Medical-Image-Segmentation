from LiverTumorSegmentation import logger
from LiverTumorSegmentation.components.evaluation import SegmentationEvaluator
from LiverTumorSegmentation.config.configuration import ConfigurationManager

STAGE_NAME = "Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_default_eval_config()
        evaluation = SegmentationEvaluator(eval_config)
        evaluation.evaluate()
        evaluation.save_score()
        evaluation.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>  {STAGE_NAME} started <<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>  {STAGE_NAME} completed <<<<<<<\n\n x===========x")
    except Exception as e:
        logger.exception(e)
        raise e