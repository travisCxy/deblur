from . import common
from . import dataset
from . import model_uni
from . import evaluator


# def get_model_configs():
#     return {
#         "model": model_uni.UniModel(),
#         "input_fn": dataset.create_dataset,
#         "train_config": {
#             "opt": "adam",
#             "lr": [0.00001, 0.000001],
#             "decay_steps": [300000 * 8, ],
#             "decay_name": "manual",
#             "max_step": 500000 * 8,
#             "suggest_gpu_num": 8
#         },
#         "evaluator": evaluator.Evaluator()
#     }
def get_model_configs():
    return {
        "model": model_uni.UniModel(),
        "input_fn": dataset.create_dataset,
        "train_config": {
            "opt": "adam",
            "lr": [0.00001, 0.000001],
            "decay_steps": [300000 * 8, ],
            "decay_name": "manual",
            "max_step": 500000 * 8,
            "suggest_gpu_num": 8
        },
        "evaluator": evaluator.Evaluator()
    }

# def get_model_configs():
#     return {
#         "model": model_uni.UniModel(),
#         "input_fn": dataset.create_dataset,
#         "train_config": {
#             "opt": "adam",
#             "lr": [0.0001, 0.00001, 0.000001],
#             "decay_steps": [200000 * 2, 300000*2],
#             "decay_name": "manual",
#             "max_step": 400000 * 2,
#             "suggest_gpu_num": 4
#         },
#         "evaluator": evaluator.Evaluator()
#     }