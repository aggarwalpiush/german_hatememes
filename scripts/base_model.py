import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
import warnings

import numpy as np
import pandas as pd
import pandas_path  # Path style access for pandas
from tqdm import tqdm


import torch
import torchvision
import fasttext

data_dir = Path.cwd().parent / "data"

img_tar_path = data_dir / "img"
train_path = data_dir / "train.jsonl"
dev_path = data_dir / "dev_seen.jsonl"
test_path = data_dir / "dev_seen_deutsch.jsonl"


train_samples_frame = pd.read_json(train_path, lines=True)


from PIL import Image


images = [
    Image.open(
        data_dir / train_samples_frame.loc[i, "img"]
    ).convert("RGB")
    for i in range(5)
]



# define a callable image_transform with Compose
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor()
    ]
)

# convert the images and prepare for visualization.
tensor_img = torch.stack(
    [image_transform(image) for image in images]
)
grid = torchvision.utils.make_grid(tensor_img)



from vl_model import HatefulMemesModel

hparams = {

    # Required hparams
    "train_path": train_path,
    "dev_path": dev_path,
    "img_dir": data_dir,

    # Optional hparams
    "embedding_dim": 150,
    "language_feature_dim": 300,
    "vision_feature_dim": 300,
    "fusion_output_size": 256,
    "output_path": "model-outputs",
    "dev_limit": None,
    "lr": 0.00005,
    "max_epochs": 10,
    "n_gpu": 1,
    "batch_size": 16,
    # allows us to "simulate" having larger batches
    "accumulate_grad_batches": 16,
    "early_stop_patience": 3
}




#hateful_memes_model = HatefulMemesModel(hparams=hparams)
#hateful_memes_model.fit()

checkpoints = list(Path("model-outputs").glob("*.ckpt"))
assert len(checkpoints) == 1

print(checkpoints)

hateful_memes_model = HatefulMemesModel.load_from_checkpoint('model-outputs/epoch=0.ckpt')
submission = hateful_memes_model.make_submission_frame(
    test_path
)
submission.to_csv(("model-outputs/german_test_baseline.csv"), index=True)
print(submission.head())
