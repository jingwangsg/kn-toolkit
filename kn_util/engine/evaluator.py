import torch
from loguru import logger

from ..utils.logger import MetricLogger
from ..data.collection_ops import nested_to, groupby
from ..dist import all_gather_object, is_main_process, synchronize


class Evaluator:

    def evaluate_batch(self, model, batch):
        """
        Return:
            result_list: List[Dict], each dict contains the intermediate results of inference and evaluation
        """
        raise NotImplementedError

    def preprocess_batch(self, batch):
        return batch

    def calculate_metrics(self, result_list):
        raise NotImplementedError

    @classmethod
    def setup_metric_logger(self):
        metric_logger = MetricLogger()
        return metric_logger

    @torch.no_grad()
    def evaluate(
        self,
        model,
        dataloader,
        eval_key="",
        log_freq=10,
        save_predictions=None,
    ):
        model.eval()
        self.model = model
        metric_logger = self.setup_metric_logger()
        result_list = []

        for i, batch in enumerate(metric_logger.log_every(dataloader, log_freq=log_freq, title=eval_key)):
            batch = nested_to(batch, device="cuda", non_blocking=True)
            batch = self.preprocess_batch(batch)
            assert "_id" in batch, "batch should contain _id field"
            result_item_batch = self.evaluate_batch(model, batch)
            result_list += result_item_batch

        # gather result_list
        _result_list_gathered = all_gather_object(result_list)

        result_list = [item for sublit in _result_list_gathered for item in sublit]
        result_dict = groupby(result_list, "_id")

        # save_predictions
        if save_predictions is not None:
            torch.save(result_dict, save_predictions)

        # calculate metrics
        metrics = self.calculate_metrics(list(result_dict.values()))
        metric_str = "\t".join([eval_key] + [f"{k}: {v:.4f}" for k, v in metrics.items()])

        logger.info(metric_str)

        return metrics
