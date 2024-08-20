from . import common
from . import dataset
from . import model_ch
from . import evaluator


def get_model_configs():
    return {
        "model": model_ch.CHModel(),
        "input_fn": dataset.create_dataset,
        "train_config": {
            "opt": "adam",
            "lr": [0.0001, 0.00001],
            "decay_steps": [400000 * 4, ],
            "decay_name": "manual",
            "max_step": 3200000,
            "suggest_gpu_num": 4
        },
        "evaluator": evaluator.Evaluator(),
    }