# all non-standard-lib imports are ignored since the pylance extension does not understand mamba env
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import pandas as pd  # type: ignore
from PIL import Image
import json
import numpy as np  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore


class JsonImageTextDataset(Dataset):
    """
    Dataset implementation for the Google Docci dataset which contains images and text descriptions.
    The descriptions and image file names for this dataset are stored in a jsonlines file,
    so each image must be fetched from disk and each text description must be encoded
    using a SentenceTransformer model.

    Inherits functionality from torch.utils.data.Dataset.

    Each item in the dataset is a tuple of a single image and a single text embedding vector.
    Images are (1, 3, 768, 768) and text embeddings are (1, 768).

    Usage:

    dataset = JsonImageTextDataset("train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for image, text in dataloader:
        print(image.shape, text.shape)

    """

    def __init__(self, split, transform=None):
        self.split = split
        self.transform = transform
        self.df = self.load(split)
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def load(self, split):
        df = pd.DataFrame(columns=["example_id", "split", "image_file", "description"])

        with open("data/descriptions.jsonlines", "r") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                row = {
                    "example_id": data["example_id"],
                    "split": data["split"],
                    "image_file": data["image_file"],
                    "description": data["description"],
                }
                df = pd.concat(
                    [df, pd.DataFrame([row])], ignore_index=True
                ).reset_index(drop=True)

        df = df[df["split"] == split].reset_index(drop=True)

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.load_image(row["image_file"])
        text = self.encode_text(row["description"])
        return image, text
        # return self.combine_image_text(image, text)

    def load_image(self, image_file):
        img = Image.open(f"data/images/{image_file}").convert("RGB").resize((768, 768))
        img = np.array(img) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).float()
        return img  # (1, 3, 768, 768), scaled to [-1, 1]

    def encode_text(self, text):
        return torch.tensor(self.model.encode(text)).unsqueeze(0)  # (1, 768)

    def combine_image_text(self, image, text):
        return torch.cat([image, text], dim=1)
