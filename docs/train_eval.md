# Prerequisites

**Please ensure you have prepared the environment and the PianoMotion10M dataset.**

# Train and Test

Train PianoMotion10M Position Predictor with Hubert and transformer. Feel free to change position feature extractor to mamba decoder by `--encoder_type mamba --num_layer 16`, and change audio feature extractor by `--wav2vec_path`. The result will be stored in `./logs/`.
```
# Base model
python train.py --experiment_name piano2posi_base --bs_dim 6 --adjust --is_random --up_list 1467634 66685747 \
 --data_root /path/to/PianoMotion10M_Dataset --iterations 100000 --batch_size 32 --train_sec 8 --feature_dim 512 \
 --wav2vec_path ./checkpoints/hubert-base-ls960 --check_val_every_n_iteration 1000 --save_every_n_iteration 1000 \
 --latest_layer tanh --encoder_type transformer --num_layer 4

# Large model
python train.py --experiment_name piano2posi_large --bs_dim 6 --adjust --is_random --up_list 1467634 66685747 \
 --data_root /path/to/PianoMotion10M_Dataset --iterations 100000 --batch_size 20 --train_sec 8 --feature_dim 512 \
 --wav2vec_path ./checkpoints/hubert-large-ls960-ft --check_val_every_n_iteration 1000 --save_every_n_iteration 1000 \
 --latest_layer tanh --encoder_type transformer --num_layer 8
```

Train PianoMotion10M Gesture Generator with Hubert and transformer. Feel free to change gesture feature extractor to mamba decoder by `--encoder_type mamba --num_layer 8(base)/16(large)`. The result will be stored in `./logs/`.

```
# Base model
python train_diffusion.py --experiment_name piano2motion_base --is_random --unet_dim 256 --iterations 100000 \
--bs_dim 96  --batch_size 32 --train_sec 8 --data_root /path/to/PianoMotion10M_Dataset \
--xyz_guide  --check_val_every_n_iteration 1000 --save_every_n_iteration 1000 \
 --adjust --piano2posi_path logs/piano2posi_base --encoder_type transformer --num_layer 4

# Large model
python train_diffusion.py --experiment_name piano2motion_large --is_random --unet_dim 256 --iterations 100000 \
--bs_dim 96  --batch_size 20 --train_sec 8  --data_root /path/to/PianoMotion10M_Dataset\
--xyz_guide  --check_val_every_n_iteration 1000 --save_every_n_iteration 1000 \
 --adjust --piano2posi_path logs/piano2posi_large  --encoder_type transformer --num_layer 8
```

Eval PianoMotion10M after training PianoMotion10M Gesture Generator on the validation set.
```
python eval.py --exp_path /path/to/logs (e.g. ./logs/piano2motion_base)  --data_root /path/to/PianoMotion10M_Dataset --valid_batch_size 64 --mode valid
```

# Visualization 

Visualize the results, which will be stored in `./results`.

```
python infer.py --exp_path /path/to/logs --data_root /path/to/PianoMotion10M_Dataset --valid_batch_size 64 --mode valid

```