from data import JsonImageTextDataset
from model_old import GaussianNoiseScheduler, UNet

import os
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = JsonImageTextDataset("train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

scheduler = GaussianNoiseScheduler(200, start=0.0001, end=0.03)

if os.path.exists("mess"):
    shutil.rmtree("mess")
    
os.makedirs("mess")

model = UNet().to(device)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

samples_per_image = 3

run = wandb.init(project="image-text-noise")

num_epochs = 1
optimizer.zero_grad()
for epoch in range(num_epochs):
    avg_loss = 0
    for i, (image, text) in tqdm(enumerate(dataloader), total=len(dataloader)):
                
        t = torch.randint(1, scheduler.T, (samples_per_image,)).type(torch.int64) # get several random timesteps
        for j in range(len(t)):
            noisy_img, noise = scheduler.forward_sample(image, torch.tensor([t[j]]))
            
            noisy_img = noisy_img.to(device)
            noise = noise.to(device)
            
            out = model(noisy_img)
            
            loss = loss_fn(out, noise)
            
            loss.backward()
            
            optimizer.step()
            
            optimizer.zero_grad()
            
            wandb.log({"loss": loss.item(), "img": i, "timestep": t[j]})
            
            avg_loss += loss.item()
            
            if (i % 100 == 0 and i != 0):
                # generate an image by iteratively removing noise from the random noise
                with torch.no_grad():
                    rand_noise = torch.randn(1, 3, 768, 768).to(device)

                    for k in range(scheduler.T):
                        pred_noise = model(rand_noise)
                        rand_noise = rand_noise - pred_noise
                    rand_noise = (rand_noise - rand_noise.min()) / (rand_noise.max() - rand_noise.min())
                    plt.imsave(f"mess/{i}.png", rand_noise.squeeze().permute(1, 2, 0).cpu().detach().numpy())
                
    wandb.log({"epoch": epoch, "avg_loss": avg_loss / (len(dataloader) * samples_per_image)})
    
    print(f"Epoch {epoch}, avg loss: {avg_loss / (len(dataloader) * samples_per_image)}")
    
run.finish()

torch.save(model.state_dict(), "model.pt")

# use random noise to generate something

noise = torch.randn(1, 3, 768, 768).to(device)
pred_noise = model(noise)

img = noise - pred_noise

img = (img - img.min()) / (img.max() - img.min())

plt.imsave("generated_image.png", img.squeeze().permute(1, 2, 0).cpu().detach().numpy())


            
        
        
    
    