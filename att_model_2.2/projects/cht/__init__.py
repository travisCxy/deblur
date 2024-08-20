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
# def get_model_configs():
#     return {
#         "model": model_uni.UniModel(),
#         "input_fn": dataset.create_dataset,
#         "train_config": {
#             "opt": "adam",
#             "lr": [0.00002, 0.000002],
#             "decay_steps": [300000 * 4, ],
#             "decay_name": "manual",
#             "max_step": 600000 * 4,
#             "suggest_gpu_num": 4
#         },
#         "evaluator": evaluator.Evaluator()
#     }

# def get_model_configs():
#     return {
#         "model": model_uni.UniModel(),
#         "input_fn": dataset.create_dataset,
#         "train_config": {
#             "opt": "adam",
#             "lr": [0.00001, 0.000001],
#             "decay_steps": [300000 * 4, ],
#             "decay_name": "manual",
#             "max_step": 900000 * 4,
#             "suggest_gpu_num": 4
#         },
#         "evaluator": evaluator.Evaluator()
#     }

# #fintune chemical
# def get_model_configs():
#     return {
#         "model": model_uni.UniModel(),
#         "input_fn": dataset.create_dataset,
#         "train_config": {
#             "opt": "adam",
#             "lr": [ 0.000001,0.0000001],
#             "decay_steps": [300000 * 4 ],
#             "decay_name": "manual",
#             "max_step": 500000 * 4,
#             "suggest_gpu_num": 4
#         },
#         "evaluator": evaluator.Evaluator()
#     }

#finetune latex
def get_model_configs():
    return {
        "model": model_uni.UniModel(),
        "input_fn": dataset.create_dataset,
        "train_config": {
            "opt": "adam",
            "lr": [ 0.000001,0.0000001],
            "decay_steps": [9000000 * 8 ],
            "decay_name": "manual",
            "max_step": 1000000 * 8,
            "suggest_gpu_num": 8
        },
        "evaluator": evaluator.Evaluator()
    }

