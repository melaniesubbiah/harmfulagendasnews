from allennlp.predictors.predictor import Predictor
from typing import List
from allennlp.common.util import JsonDict, sanitize

from allennlp.data import Instance

@Predictor.register("rationale_predictor")
class RationalePredictor(Predictor) :
    def _json_to_instance(self, json_dict):
        # import pydevd_pycharm
        # pydevd_pycharm.settrace('160.39.167.101', port=7777, stdoutToServer=True,
        #                         stderrToServer=True)

        raise NotImplementedError

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        # import pydevd_pycharm
        # pydevd_pycharm.settrace('160.39.167.101', port=7777, stdoutToServer=True,
        #                         stderrToServer=True)
        self._model.prediction_mode = True
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

