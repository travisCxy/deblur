from . import common
from . import dataset
from . import model_sl
from . import evaluator


def get_model_configs():
    return {
        "model": model_sl.SlModel(),
        "input_fn": dataset.create_dataset,
        "train_config": {
            "opt": "adam",
            "lr": [0.0001, 0.00001 ],
            "decay_steps": [200000 * 4, ], #[12000 * 2, 24000 * 2],
            "decay_name": "manual",
            "max_step": 600000 * 4,
            "suggest_gpu_num": 4
        },
        "evaluator": evaluator.Evaluator()
    }