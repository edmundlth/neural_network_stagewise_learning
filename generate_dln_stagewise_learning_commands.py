import numpy as np
import datetime
import sys
from utils import to_json_friendly_tree
import itertools
import json


def _create_cmd_string(key, value):
    return f'"{key}={value}"'
    if isinstance(value, str):
        return f"{key}='{value}'"
    elif isinstance(value, (list, tuple)):
        val_str = []
        for v in value:
            if isinstance(v, str):
                val_str.append(f"'{v}'")
            else:
                val_str.append(str(v))
        val_str = ",".join(val_str)
        return f'{key}="{val_str}"'
    else:
        return f"{key}={value}"
    
def generate_sacred_commands(fixed_configs, varying_configs, script_name, observer=None):
    if observer is None:
        observer = ""
    prefix_string = f"python {script_name} {observer} with"
    keys, values = zip(*varying_configs.items())
    commands = []
    for combo in itertools.product(*values):
        cmd = [prefix_string]

        for key, value in fixed_configs.items():
            cmd.append(_create_cmd_string(key, value))

        for key, value in zip(keys, combo):
            cmd.append(_create_cmd_string(key, value))

        commands.append(" ".join(cmd))
    return commands


current_time = datetime.datetime.now()
datetime_str = current_time.strftime("%Y%m%d%H%M")

DLN_SIZE = "small"
IN_DIM = 5
OUT_DIM = 5
NUM_HIDDEN_LAYERS_MIN, NUM_HIDDEN_LAYERS_MAX = 1, 5
WIDTH_TYPE = "vary" # "vary", "constant"
WIDTH_MIN, WIDTH_MAX = 5, 10
WIDTH = min(IN_DIM, OUT_DIM)
assert min(IN_DIM, OUT_DIM) <= WIDTH_MIN <= WIDTH_MAX
assert min(IN_DIM, OUT_DIM) <= WIDTH

NUMTRAININGDATA = 100000
BATCH_SIZE = 512
LEARNING_RATE = 1e-5
NUMSTEPS = 500000
OPTIM = "sgd"
NUM_SEEDS = 1


if WIDTH_TYPE == "vary":
    width_str = f"{WIDTH_MIN}-{WIDTH_MAX}"
    width_options = [
        np.random.randint(WIDTH_MIN, WIDTH_MAX+1, size=ell).tolist()
        for ell in range(NUM_HIDDEN_LAYERS_MIN, NUM_HIDDEN_LAYERS_MAX+1)
    ]
elif WIDTH_TYPE == "constant":
    width_str = f"{WIDTH}"
    width_options = [[WIDTH] * i for i in range(NUM_HIDDEN_LAYERS_MIN, NUM_HIDDEN_LAYERS_MAX+1)]
else:
    raise ValueError(f"Invalid WIDTH_TYPE: {WIDTH_TYPE}")

EXPT_NAME = (
    f"dln_{DLN_SIZE}_"
    f"width{width_str}_"
    f"layers{NUM_HIDDEN_LAYERS_MIN}-{NUM_HIDDEN_LAYERS_MAX}_"
    f"in{IN_DIM}_out{OUT_DIM}_"
    f"n{NUMTRAININGDATA}_"
    f"bs{BATCH_SIZE}_lr{LEARNING_RATE}_nstep{NUMSTEPS}_optim{OPTIM}_"
    f"{datetime_str}"
)

DB_NAME = "test_dln_stagewise_learning"
# DB_NAME = "dln_stagewise_learning"
# SACRED_OBSERVER = f"-m localhost:27017:{DB_NAME}"
SACRED_OBSERVER = f"-F ./outputs/dln_stagewise_learning/{EXPT_NAME}/"


# Base parameters (constant across all runs)
FIXED_CONFIGS = {
    "expt_name": EXPT_NAME,
    "use_behavioural": True,
    "do_llc_estimation": False,
    "loss_trace_minibatch": True,
    "burn_in_prop": 0.9,
    "logging_period": 100,
    "log_full_checkpoint_param": False,
    "verbose": True, 
    "data_config.num_training_data": NUMTRAININGDATA,
    "data_config.output_noise_std": 0.1,
    "data_config.input_variance_range": [1.0, 10],
    "model_config.input_dim": IN_DIM,
    "model_config.output_dim": OUT_DIM,
    "training_config.learning_rate": LEARNING_RATE,
    "training_config.batch_size": 512,
    "training_config.num_steps": NUMSTEPS,
    "training_config.optim": OPTIM,
    "training_config.momentum": None,
    "training_config.early_stopping_loss_threshold": 0.0099,
}

# Parameters to vary (list of possible values for that parameter)
VARYING_CONFIGS = {
    "seed": list(range(NUM_SEEDS)),
    "data_config.teacher_matrix": [
        ("diagonal", 50, 10), 
        ("diag_power_law", 1.0, 50)
    ],
    "data_config.idcorr": [True, False],
    "model_config.hidden_layer_widths": width_options,
    "model_config.initialisation_exponent": [-3.0, -1.0, 1.0, 3.0], 
    
}

# Generate commands
COMMANDS = generate_sacred_commands(
    FIXED_CONFIGS, 
    VARYING_CONFIGS, 
    script_name="expt_dln_stagewise_learning.py",
    observer=SACRED_OBSERVER
)
print(EXPT_NAME)
print(f"Num commands: {len(COMMANDS)}")

# Save the commands to a file
if len(sys.argv) > 1:
    filepath = sys.argv[1]
else: 
    filepath = f"./outputs/dln_stagewise_learning/commands_{datetime_str}.txt"

print(f"Saving commands to {filepath}")
with open(filepath, "w") as outfile:
    header_str = to_json_friendly_tree({
        "expt_name": EXPT_NAME,
        "fixed_configs": FIXED_CONFIGS,
        "varying_configs": VARYING_CONFIGS
    })
    header_str = json.dumps(header_str, indent=None)
    outfile.write(header_str)
    outfile.write('\n'.join(COMMANDS))
