# Imports
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
# Pipeline Imports
import infer_config_updated
from UnpairedDataset import UnpairedDataset
from PairedDataset import PairedDataset
from cyclegan_models import CycleGan
from utils import save_paired_image, save_unpaired_image, load_weights, convert_to_255

import warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

if __name__ == "__main__":
    # Get the config file
    config = infer_config_updated.config 
    root = config['data_path']
    mode = config['sub_fold']

    # Set paired or unpaired
    if config['paired']:
        test_ds = PairedDataset(root, mode)
    else:
        test_ds = UnpairedDataset(root, mode)

    # Create Dataloader
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    # Iterate through checkpoints
    for ckpt_name in config["ckpt_names"]:
        # Model Loaded
        print(f"Loading model from -> {os.path.join(config['model_name'], ckpt_name)}\n")
        model = CycleGan.load_from_checkpoint(os.path.join(config['model_name'], ckpt_name))

        # Set model to 'cuda' and evaluation mode
        model.eval()
        model.to('cuda')

        # Extract epoch number for folder naming
        epoch_number = ckpt_name.split('=')[1].split('-')[0]

        # Iterate through batches
        for i, batch in enumerate(tqdm(test_dl, total=len(test_dl))):
            imgA, imgB = batch['A'].to('cuda'), batch['B'].to('cuda')        
            with torch.no_grad():
                fakeB = model.genX(imgA)
                fakeA = model.genY(imgB)

            # Convert back to cpu for saving image
            imgA, imgB, fakeB, fakeA = imgA.cpu(), imgB.cpu(), fakeB.cpu(), fakeA.cpu()
            pathA, pathB = batch['pathA'], batch['pathB']
            for i in range(len(pathA)):
                temp_pathA = os.path.split(pathA[i])[-1]
                temp_pathB = os.path.split(pathB[i])[-1]
                if config['paired']:
                    path_dir = os.path.join(config['model_name'], f'Epoch_{epoch_number}_{config["sub_fold"]}_Predictions')
                    os.makedirs(path_dir, exist_ok=True)
                    path = os.path.join(path_dir, temp_pathA)
                    save_paired_image(imgA, imgB, fakeB, fakeA, path)
                else:
                    # For A
                    path_dir = os.path.join(config['model_name'], f'Epoch_{epoch_number}_{config["sub_fold"]}_Predictions_A')
                    os.makedirs(path_dir, exist_ok=True)
                    path = os.path.join(path_dir, temp_pathA)
                    save_unpaired_image(imgA, fakeB, path)
                    # For B 
                    path_dir = os.path.join(config['model_name'], f'Epoch_{epoch_number}_{config["sub_fold"]}_Predictions_B')
                    os.makedirs(path_dir, exist_ok=True)
                    path = os.path.join(path_dir, temp_pathB)
                    save_unpaired_image(imgB, fakeA, path)

        print(f"Inference Completed for epoch {epoch_number}!")
