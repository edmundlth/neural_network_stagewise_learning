import numpy as np
import datetime
import sys
from utils import to_json_friendly_tree
import json
from utils import write_commands_to_file, generate_sacred_commands


current_time = datetime.datetime.now()
datetime_str = current_time.strftime("%Y%m%d%H%M")

NAME="largelong"
NUM_HIDDEN_LAYERS_MIN, NUM_HIDDEN_LAYERS_MAX = 1, 1
WIDTH_TYPE = "single" # "vary", "constant", single
WIDTH_MIN, WIDTH_MAX = 500, 3000
WIDTH = 1000
NUM_WIDTH_VARIATION = 5
assert WIDTH_MIN <= WIDTH_MAX
if WIDTH_TYPE == "vary":
    width_str = f"vary-{WIDTH_MIN}-{WIDTH_MAX}"
    width_options = [
        np.random.randint(WIDTH_MIN, WIDTH_MAX+1, size=ell).tolist()
        for ell in range(NUM_HIDDEN_LAYERS_MIN, NUM_HIDDEN_LAYERS_MAX+1)
    ]
elif WIDTH_TYPE == "constant":
    width_str = f"const-{WIDTH}"
    width_options = [[WIDTH] * i for i in range(NUM_HIDDEN_LAYERS_MIN, NUM_HIDDEN_LAYERS_MAX+1)]
elif WIDTH_TYPE == "single":
    width_str = f"single-{WIDTH_MIN}-{WIDTH_MAX}-{NUM_WIDTH_VARIATION}"
    width_options = [[w] for w in np.linspace(WIDTH_MIN, WIDTH_MAX, num=NUM_WIDTH_VARIATION, dtype=int)]
else:
    raise ValueError(f"Invalid WIDTH_TYPE: {WIDTH_TYPE}")
width_options = [str(width_list) for width_list in width_options] # to fix sacred list parsing issue


NUMTRAININGDATA = 1000000
BATCH_SIZE = 20000
LEARNING_RATE = 1e-3
NUMSTEPS = 15000000
OPTIM = "adam" # "sgd", "adam", "momentum"

DO_LLC_ESTIMATION = True
LOG_SGLD_LOSS_TRACE = False
NUM_SEEDS = 1


EXPT_NAME = (
    f"multitasksparseparity_{NAME}_"
    f"width{width_str}_"
    f"layers{NUM_HIDDEN_LAYERS_MIN}-{NUM_HIDDEN_LAYERS_MAX}_"
    f"n{NUMTRAININGDATA}_"
    f"bs{BATCH_SIZE}_lr{LEARNING_RATE}_nstep{NUMSTEPS}_optim{OPTIM}_"
    f"llc{DO_LLC_ESTIMATION}_trace{LOG_SGLD_LOSS_TRACE}_"
    f"{datetime_str}"
)

is_prod = True
dev_str = "prod/" if is_prod else "dev/"
SACRED_OBSERVER = f"-F ./outputs/multitask_sparse_parity_expt/{dev_str}{EXPT_NAME}/"


# Base parameters (constant across all runs)
FIXED_CONFIGS = {
    "expt_name": EXPT_NAME,
    "do_llc_estimation": DO_LLC_ESTIMATION,
    "burn_in_prop": 0.9,
    "logging_period": 50000,
    "log_space_uniform_logging": True,
    "log_sgld_loss_trace": LOG_SGLD_LOSS_TRACE,
    "verbose": True, 

    "data_config.num_training_samples": NUMTRAININGDATA,
    "data_config.num_test_samples": NUMTRAININGDATA // 5,

    "model_config.model_type": "mlp",

    "training_config.learning_rate": LEARNING_RATE,
    "training_config.batch_size": BATCH_SIZE,
    "training_config.num_steps": NUMSTEPS,
    "training_config.optim": OPTIM,
    "training_config.momentum": .9,
    "training_config.early_stopping_loss_threshold": 1e-10,

    "sgld_config.gamma": 1.0,
    "sgld_config.num_chains": 1,
    "sgld_config.batch_size": None,

    "do_taskwise_training": True,
    "max_num_stages": 20,
}

# Parameters to vary (list of possible values for that parameter)
VARYING_CONFIGS = {
    "seed": list(range(NUM_SEEDS)),

    "data_config.n_tasks": [500],
    "data_config.n_taskbits": [100],
    "data_config.n_subset_size": [3],
    "data_config.alpha": [0.4],

    "model_config.hidden_layer_widths": width_options,

    "sgld_config.epsilon": [2e-6],
    "sgld_config.num_steps": [2500],

    "taskwise_training_num_steps": [200000]
}

# Generate commands
COMMANDS = generate_sacred_commands(
    FIXED_CONFIGS, 
    VARYING_CONFIGS, 
    script_name="expt_multitask_sparse_parity.py",
    observer=SACRED_OBSERVER
)
print(EXPT_NAME)
print(f"Num commands: {len(COMMANDS)}")

# Save the commands to a file
if len(sys.argv) > 1:
    filepath = sys.argv[1]
else: 
    filepath = f"./outputs/multitask_sparse_parity_expt/commands_multitasksparseparity_{datetime_str}.txt"

header_str = to_json_friendly_tree({
    "expt_name": EXPT_NAME,
    "fixed_configs": FIXED_CONFIGS,
    "varying_configs": VARYING_CONFIGS
})
header_str = json.dumps(header_str, indent=None)
write_commands_to_file(filepath, COMMANDS, header_str)