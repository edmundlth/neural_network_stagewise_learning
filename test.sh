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
# python expt_dln_stagewise_learning.py -m ${DATABASE_URL} with expt_name="test_run" log_full_checkpoint_param=False model_config.initialisation_exponent=-2 model_config.input_dim=5 model_config.output_dim=5 model_config.hidden_layer_widths=5,5 logging_period=300 seed=0 training_config.num_steps=30000



###########################################
# Generate commands for running on cluster
###########################################
# Read suffix from next argument 
# SUFFIX=$1
# if [ -z "$SUFFIX" ]
# then
#     echo "No suffix provided. Exiting."
#     exit 1
# fi

# OUTFILEPATH="./generated_commands_dir/commands_${SUFFIX}.txt"
# echo "Outputting commands to ${OUTFILEPATH}"
# python generate_dln_stagewise_learning_commands.py ${OUTFILEPATH}


###########################################
# Plotting commands
###########################################

FULLDATADIR=$1
OUTPUTDIR=$2

# Check if the required arguments are provided
if [ -z "$FULLDATADIR" ] || [ -z "$OUTPUTDIR" ]; then
    echo "Usage: $0 <full_data_directory> <output_directory>"
    exit 1
fi

# Ensure the output directory exists
mkdir -p "$OUTPUTDIR"

# Get the base directory and subdirectory from FULLDATADIR
DATABASEDIR=$(dirname "$FULLDATADIR")
DATASUBDIR=$(basename "$FULLDATADIR")

# Loop through all directories in FULLDATADIR
for dir in "$FULLDATADIR"/*; do
    if [ -d "$dir" ]; then
        SUBDIR=$(basename "$dir")
        
        # Check if info.json or info.json.gz exists in the directory
        if [ -f "$dir/info.json" ] || [ -f "$dir/info.json.gz" ]; then
            echo "Plotting for ${DATABASEDIR}/${DATASUBDIR}/${SUBDIR}"
            python plot_utils.py --data_dir "$DATABASEDIR" --data_subdir "${DATASUBDIR}/${SUBDIR}" --output_dir "$OUTPUTDIR"
        else
            echo "Skipping ${DATABASEDIR}/${DATASUBDIR}/${SUBDIR} - no info.json or info.json.gz found"
        fi
    fi
done

echo "Done"