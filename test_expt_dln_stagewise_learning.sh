#!/bin/bash

DATABASE_NAME="test_dln_stagewise_learning"
DATABASE_URL="localhost:27017:${DATABASE_NAME}"

#################################
# Small test
#################################
# with observer
# python expt_dln_stagewise_learning.py -m ${DATABASE_URL} with expt_name="test_run" log_full_checkpoint_param=False model_config.initialisation_exponent=2 logging_period=300 seed=0 training_config.num_steps=10000 

# no observer
# python expt_dln_stagewise_learning.py with expt_name="test_run" log_full_checkpoint_param=False model_config.initialisation_exponent=2 logging_period=300 seed=0 training_config.num_steps=10000 

#################################
# Larger test
#################################
# at origin
# python expt_dln_stagewise_learning.py -m ${DATABASE_URL} with expt_name="test_run" log_full_checkpoint_param=False model_config.initialisation_exponent=2 model_config.input_dim=5 model_config.output_dim=5 model_config.hidden_layer_widths=5,5 logging_period=300 seed=0 training_config.num_steps=100000

# away from origin
python expt_dln_stagewise_learning.py -m ${DATABASE_URL} with expt_name="test_run" log_full_checkpoint_param=False model_config.initialisation_exponent=-2 model_config.input_dim=5 model_config.output_dim=5 model_config.hidden_layer_widths=5,5 logging_period=300 seed=0 training_config.num_steps=30000

