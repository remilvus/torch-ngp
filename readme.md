# FreSh - NeRF experiments

This repository is a modified version of [torch-ngp](https://github.com/ashawkey/torch-ngp),
containing NeRF experiments testing the FreSh method. If you wany to do something else that
reproducing our experiments, please refer to the original repository. 
The main FreSh repository can be found \[here\](TODO).

## Setup

### Code

See [the original repository](https://github.com/ashawkey/torch-ngp). 
We found the original `environment.yml` file to not work well, and we provide an updated version.
We recommend installing the "pip" section of `environment.yml` after using conda to set up the environment.

### Data

Download [nerf_synthetic folder](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
and place it under `./data`

## Experiments

Below you can find the commands needed to run the experiments.

Save model outputs at initialisation (commands below assume you are using slurm array jobs):
```bash
export WORKSPACE="model_outputs"
export scale=0.8 # set to 0.6 for ship
export DATASET=lego

# In case you don't want to use slurm array jobs:
# export SLURM_ARRAY_TASK_ID=1

# Fourier
python main_nerf.py data/nerf_synthetic/$DATASET --workspace "${WORKSPACE}/fourier/${DATASET}" --ckpt="scratch"  --preload --render_color --embedding='rff' --sigma=$SLURM_ARRAY_TASK_ID  --scale $scale --num_layers=7 --num_layers_color=1 --hidden_features=256

# Siren
export OMEGA=$((10 * $SLURM_ARRAY_TASK_ID))
python main_nerf.py data/nerf_synthetic/$DATASET --workspace "${WORKSPACE}/siren/${DATASET}" --ckpt="scratch" --hidden_features=256 --preload --render_color --embedding='id' --omega=$OMEGA --activation="Sine"  --scale $scale --num_layers=7 --num_layers_color=1 --hidden_features=256

# Finer
export OMEGA=$((10 * $SLURM_ARRAY_TASK_ID))
export k=0
# if you want to optimise bias use
#export k=$(echo "scale=1; $SLURM_ARRAY_TASK_ID / 10" | bc)
python main_nerf.py data/nerf_synthetic/$DATASET --workspace "${WORKSPACE}/finer/${DATASET}" --hidden_features=256 --activation="Finer" --preload --render_color  --embedding='id' --omega=1 --finer_k=$k --omega_finer=$OMEGA --finer_high_values  --scale $scale --num_layers=7 --num_layers_color=1

# Hashgrid
# Feature vectors have to be initialised with high values (`--hashmap_high_values`), as otherwise their influence on output is too small
python main_nerf.py data/nerf_synthetic/$DATASET --workspace "${WORKSPACE}/hashgrid/${DATASET}" --ckpt="scratch"  --preload --render_color --hashmap_high_values  --scale $scale --embedding='hashgrid' --desired_resolution=$SLURM_ARRAY_TASK_ID  --log2_hashmap_size=13  --hidden_features=64 --num_layers=1 --num_layers_color=2
```

Run the FreSh method (you need the script from the main FreSh repository):
```bash
python <path_to_fresh>/scripts/find_optimal_config.py \
  --dataset model_outputs/siren/lego.npy  \
  --model_output model_outputs/siren/lego  \
  --results_root wasserstein_results/example
```
You will find the configurations selected by FreSh in `wasserstein_results/example/wasserstein_best.csv`.
For an additional description of using the script see the main FreSh repository.

Train a NeRF model:
```bash
export ITERS=500000
export scale=0.8 # set to 0.6 for ship
export DATASET=lego
export WORKSPACE="results/${DATASET}"

# Positional encoding
python main_nerf.py data/nerf_synthetic/$DATASET --workspace "${WORKSPACE}/positional" --hidden_features=256  --cuda_ray --preload  --embedding='pos' --lr=0.001  --scale $scale  --num_layers=7 --num_layers_color=1

# Siren
export OMEGA=30
python main_nerf.py data/nerf_synthetic/$DATASET --workspace "${WORKSPACE}/siren_${OMEGA}"  --hidden_features=256 --activation="Sine" --cuda_ray --preload  --embedding='id' --omega=$OMEGA --lr=0.00005  --scale $scale --num_layers=7 --num_layers_color=1

# Fourier
export SIGMA=1.0 
python main_nerf.py data/nerf_synthetic/$DATASET --workspace "${WORKSPACE}/fourier_${SIGMA}" --hidden_features=256 --cuda_ray --preload  --embedding='rff' --sigma=$SIGMA --lr=0.001  --scale $scale --num_layers=7 --num_layers_color=1

# Hashgrid
export nmax=2048
python main_nerf.py data/nerf_synthetic/$DATASET --workspace "${WORKSPACE}/hashhrid" --hidden_features=64 --num_layers=1 --num_layers_color=2 --cuda_ray --preload  --num_rays=20480  --lr=0.001 --iters=30000 --eval_interval=50   --scale $scale --embedding='hashgrid' --desired_resolution=nmax  --log2_hashmap_size=13

# Finer
export OMEGA=30
# to disable bias set `--finer_k=0`
python main_nerf.py data/nerf_synthetic/$DATASET --workspace "${WORKSPACE}/finer" --hidden_features=256 --activation="Finer" --cuda_ray --preload  --embedding='id' --omega=1 --lr=0.00005 --scale $scale --num_layers=7 --num_layers_color=1 --omega_finer=$OMEGA
```

