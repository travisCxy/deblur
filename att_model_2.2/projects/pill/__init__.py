from . import common
from . import dataset
from . import model_pill
from . import evaluator


def get_model_configs():
    return {
        "model": model_pill.PillModel(),
        "input_fn": dataset.create_dataset,
        "train_config": {
            "opt": "adam",
            "lr": [0.0001, 0.00001],
            "decay_steps": [16000 * 1, ],
            "decay_name": "manual",
            "max_step": 32000,
            "suggest_gpu_num": 1
        },
        "evaluator": evaluator.Evaluator(),
    }
