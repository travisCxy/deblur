from . import common
from . import dataset
from . import model_ml
from . import evaluator


def get_model_configs():
    return {
        "model": model_ml.MlModel(),
        "input_fn": dataset.create_dataset,
        "train_config": {
            "opt": "adam",
            "lr": [0.0001, 0.00001],
            "decay_steps": [300000 * 4, ],
            "decay_name": "manual",
            "max_step": 500000 * 8,
            "suggest_gpu_num": 8
        },
        "evaluator": evaluator.Evaluator(),
        "weight_decay": 0.000004,
    }
