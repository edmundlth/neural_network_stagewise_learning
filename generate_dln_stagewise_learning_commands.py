import numpy as np
import datetime
import sys
from utils import to_json_friendly_tree
import json
from utils import write_commands_to_file, generate_sacred_commands


current_time = datetime.datetime.now()
datetime_str = current_time.strftime("%Y%m%d%H%M")

NAME="banded"
DLN_SIZE = "large"
IN_DIM = 256
OUT_DIM = 256
NUM_HIDDEN_LAYERS_MIN, NUM_HIDDEN_LAYERS_MAX = 3, 4
WIDTH_TYPE = "vary" # "vary", "constant"
WIDTH_MIN, WIDTH_MAX = min(IN_DIM, OUT_DIM), 800
WIDTH = min(IN_DIM, OUT_DIM)
assert min(IN_DIM, OUT_DIM) <= WIDTH_MIN <= WIDTH_MAX
assert min(IN_DIM, OUT_DIM) <= WIDTH

NUMTRAININGDATA = 100000
BATCH_SIZE = 1024
LEARNING_RATE = 2e-4
NUMSTEPS = 800000
OPTIM = "momentum" # "sgd", "adam", "momentum"

DO_LLC_ESTIMATION = True
NUM_SEEDS = 3


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
    f"dln_{NAME}_{DLN_SIZE}_"
    f"width{width_str}_"
    f"layers{NUM_HIDDEN_LAYERS_MIN}-{NUM_HIDDEN_LAYERS_MAX}_"
    f"in{IN_DIM}_out{OUT_DIM}_"
    f"n{NUMTRAININGDATA}_"
    f"bs{BATCH_SIZE}_lr{LEARNING_RATE}_nstep{NUMSTEPS}_optim{OPTIM}_"
    f"llc{DO_LLC_ESTIMATION}_"
    f"{datetime_str}"
)

# DB_NAME = "test_dln_stagewise_learning"
# SACRED_OBSERVER = f"-m localhost:27017:{DB_NAME}"
is_prod = True
dev_str = "prod/" if is_prod else "dev/"
SACRED_OBSERVER = f"-F ./outputs/dln_stagewise_learning/{dev_str}{EXPT_NAME}/"


# Base parameters (constant across all runs)
FIXED_CONFIGS = {
    "expt_name": EXPT_NAME,
    "use_behavioural": True,
    "do_llc_estimation": DO_LLC_ESTIMATION,
    "loss_trace_minibatch": True,
    "burn_in_prop": 0.9,
    "logging_period": 1200,
    "log_full_checkpoint_param": False,
    "verbose": True, 
    "data_config.num_training_data": NUMTRAININGDATA,
    "data_config.output_noise_std": 0.1,
    "data_config.input_variance_range": [0.5, 2.5], # P S P^T
    "model_config.input_dim": IN_DIM,
    "model_config.output_dim": OUT_DIM,
    "training_config.learning_rate": LEARNING_RATE,
    "training_config.batch_size": BATCH_SIZE,
    "training_config.num_steps": NUMSTEPS,
    "training_config.optim": OPTIM,
    "training_config.momentum": .9,
    "training_config.early_stopping_loss_threshold": 0.0099,
    "sgld_config.gamma": 1.0,
    "sgld_config.num_chains": 5,
    "sgld_config.batch_size": 512,
}

# Parameters to vary (list of possible values for that parameter)
VARYING_CONFIGS = {
    "seed": list(range(NUM_SEEDS)),
    "data_config.teacher_matrix": [
        {"type": "band_power_law", "config_vals": "[5.0,1.05,3,300]"},
        {"type": "band_power_law", "config_vals": "[10.0,1.01,3,300]"},
        # {"type": "band", "config_vals": "[2.0,20.0,2]"},
        # {"type": "band", "config_vals": "[1.0,5.0,2]"},
        # {"type": "band", "config_vals": "[1.0,10.0,2]"},
    ],
    "data_config.idcorr": [True],
    "model_config.hidden_layer_widths": width_options,
    "model_config.initialisation_exponent": [2.0], # [-0.3, -0.1, 1.0, 2.0], # Jocot, 2019
    "sgld_config.epsilon": [2e-8],
    "sgld_config.num_steps": [2000],
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

header_str = to_json_friendly_tree({
    "expt_name": EXPT_NAME,
    "fixed_configs": FIXED_CONFIGS,
    "varying_configs": VARYING_CONFIGS
})
header_str = json.dumps(header_str, indent=None)
write_commands_to_file(filepath, COMMANDS, header_str)