from . import common
from . import dataset
from . import model_hw
from . import evaluator


def get_model_configs():
    return {
        "model": model_hw.HWModel(),
        "input_fn": dataset.create_dataset,
        "train_config": {
            "opt": "adam",
            "lr": [0.0001, 0.00001, 0.000001],
            "decay_steps": [120000 * 4, 240000 * 4],
            "decay_name": "manual",
            "max_step": 1000000,
            "suggest_gpu_num": 4
        },
        "evaluator": evaluator.Evaluator()
    }