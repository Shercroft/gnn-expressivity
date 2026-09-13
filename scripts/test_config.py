from gnn_expressivity.training.config import (
    load_yaml,
    merge_configs,
)

model_config = load_yaml(
    "configs/models/gin.yaml"
)

encoding_config = load_yaml(
    "configs/encodings/rwse.yaml"
)

task_config = load_yaml(
    "configs/tasks/brec.yaml"
)

config = merge_configs(
    model_config,
    encoding_config,
    task_config,
)

print("MERGED CONFIG")
print(config)
