# RLDIF

Welcome to RLDIF! This README contains detailed instructions for configuring and running RLDIF both locally and in a Docker instance. Please read through carefully to ensure successful execution.

## Config File Instructions

To begin, modify the config file located in the `config` directory. Only change the parameters listed below. **Do not alter any other part of the config file.**

### Parameters

- **name** (str):  
  The name of the result file when running in inference mode.

- **inference** (bool):  
  Set to `true` for inference mode or `false` for training mode.

- **protein_mpnn** (bool):  
  Enable or disable running inference with ProteinMPNN.

- **rldif** (bool):  
  Enable or disable running inference or finetuning with RLDIF.

- **esmif** (bool):  
  Enable or disable running inference with ESMIF.

- **dif_large** (bool):  
  Enable or disable running inference or finetuning with DIF_Large.

- **temp** (float):  
  Set to `false` for no temperature sampling (applies to DIF_Large and RLDIF). For ProteinMPNN or ESMIF, provide a float between 0 and 1 to specify temperature sampling.

- **docker** (bool):  
  Set to `true` if running in a Docker instance; otherwise, set to `false`.

- **free_positions** (bool):  
  Specify if free positions should be considered.

- **num_samples** (int):  
  Number of samples for inference or fine-tuning.

- **custom_pdb_input**:  
  The name of the input CSV file. When running a regular inference script, provide the absolute path. Refer to Dockerfile instructions for running within a Docker file.

- **data.train_pdb_input**:
  Path to the input train CSV file. Must be defined for finetuning. The CSV contains two columns one for absolute path of pdb file and another for chain ID.

- **data.val_pdb_input**:
  Path to the input val CSV file. Must be defined for finetuning. The CSV contains two columns one for absolute path of pdb file and another for chain ID.

- **data.test_pdb_input**:
  Path to the input test CSV file. Must be defined for finetuning. The CSV contains two columns one for absolute path of pdb file and another for chain ID.

### Running Scripts

After updating the config file with the required parameters for either inference or finetuning, execute the following command from the appropriate directory:

```bash
python -m run.run
```

## Dockerfile Instructions 

Docker functionality has only been used for inference, not finetuning.

To build and run the Docker instance for inference, follow these steps:

1. **Build the Docker Image:**

   From the main directory containing the pre-build Docker, run:

   ```bash
   docker load -i inverse_folding.tar
   ```
2. **Run the Docker Instance:**

  **Directory Structure:**  
  To run inference on a set of PDB files, create a directory and ensure it contains:
  1. A config file: Copy from the `configs` directory and modify necessary parameters. Make sure to set the docker parameter to be `true`
  2. An input CSV file: This should have two columns - the absolute path to the PDB file and the chain ID(s) desired.
  3. The PDB files to be used for inference.

   
  Use the command below to run the Docker instance for inference:

  ```bash
   docker run --rm -e PDB_BASE_PATH=/usr/src/app/input_files/ -v /home/input_file_directory/:/usr/src/app/input_files/ --gpus all inverse_folding
   ```

  Ensure that you provide the absolute path in place of `/home/input_file_directory/`.

## Features to come

1. RL functionality, it has been integrated I am just testing it now.

### How to cite this work
``` @article{ektefaie2024reinforcement,
  title={Reinforcement learning on structure-conditioned categorical diffusion for protein inverse folding},
  author={Ektefaie, Yasha and Viessmann, Olivia and Narayanan, Siddharth and Dresser, Drew and Kim, J Mark and Mkrtchyan, Armen},
  journal={arXiv preprint arXiv:2410.17173},
  year={2024}}
 ```

