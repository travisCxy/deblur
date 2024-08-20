from . import common
from . import dataset
from . import model_equ
from . import evaluator


def get_model_configs():
    return {
        "model": model_equ.EquModel(),
        "input_fn": dataset.create_dataset,
        "train_config": {
            "opt": "adam",
            "lr": [0.001, 0.0001, 0.00001, 0.000001],
            "decay_steps": [60000 * 4, 120000 * 4, 160000 * 4],
            "decay_name": "manual",
            "max_step": 180000 * 4,
            "suggest_gpu_num": 4
        },
        "evaluator": evaluator.Evaluator()
    }
