import typing as tp

from regex import P
import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
import dataset
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Settings, Clothes, seed_everything
from vit import ViT

from profiler import Profile, Schedule


def get_vit_model() -> torch.nn.Module:
    model = ViT(
        depth=12,
        heads=4,
        image_size=224,
        patch_size=32,
        num_classes=20,
        channels=3,
    ).to(Settings.device)
    return model


def get_loaders() -> torch.utils.data.DataLoader:
    dataset.download_extract_dataset()
    train_transforms = dataset.get_train_transforms()
    val_transforms = dataset.get_val_transforms()

    frame = pd.read_csv(f"{Clothes.directory}/{Clothes.csv_name}")
    train_frame = frame.sample(frac=Settings.train_frac)
    val_frame = frame.drop(train_frame.index)

    train_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", train_frame, transform=train_transforms
    )
    val_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", val_frame, transform=val_transforms
    )

    print(f"Train Data: {len(train_data)}")
    print(f"Val Data: {len(val_data)}")

    train_loader = DataLoader(dataset=train_data, batch_size=Settings.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset=val_data, batch_size=Settings.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, val_loader


def run_epoch(model, train_loader, val_loader, criterion, optimizer) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0, 0
    val_loss, val_accuracy = 0, 0
    model.train()
    
    with Profile(
        model,
        schedule=Schedule(wait=0, warmup=1, active=1),
    ) as prof:
        for data, label in tqdm(train_loader, desc="Train"):
            data = data.to(Settings.device)
            label = label.to(Settings.device)
            output = model(data)
            loss = criterion(output, label)
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item() / len(train_loader)
            epoch_loss += loss.item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            prof.step()
            if prof._phase == "inactive":
                return

    model.eval()
    for data, label in tqdm(val_loader, desc="Val"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        val_accuracy += acc.item() / len(train_loader)
        val_loss += loss.item() / len(train_loader)

    return epoch_loss, epoch_accuracy, val_loss, val_accuracy

def run_epoch_torch(model, train_loader, val_loader, criterion, optimizer) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0, 0
    val_loss, val_accuracy = 0, 0
    model.train()
    
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("torch_profiler"),
    ) as prof: 
        num_iters = 2
        for data, label in tqdm(train_loader, desc="Train"):
            data = data.to(Settings.device)
            label = label.to(Settings.device)
            output = model(data)
            loss = criterion(output, label)
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item() / len(train_loader)
            epoch_loss += loss.item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            prof.step()
            if prof.step_num >= num_iters:
                return

    model.eval()
    for data, label in tqdm(val_loader, desc="Val"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        val_accuracy += acc.item() / len(train_loader)
        val_loss += loss.item() / len(train_loader)

    return epoch_loss, epoch_accuracy, val_loss, val_accuracy

def run_epoch_torch_fp16(model, train_loader, val_loader, criterion, optimizer) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0, 0
    val_loss, val_accuracy = 0, 0
    model.train()
    
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("torch_profiler"),
    ) as prof: 
        num_iters = 2
        scaler = torch.amp.GradScaler()

        for data, label in tqdm(train_loader, desc="Train"):
            data = data.to(Settings.device)
            label = label.to(Settings.device)
            with torch.amp.autocast('cuda', torch.float16):
                output = model(data)
                loss = criterion(output, label)
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item() / len(train_loader)
            epoch_loss += loss.item() / len(train_loader)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            prof.step()
            if prof.step_num >= num_iters:
                return

    model.eval()
    for data, label in tqdm(val_loader, desc="Val"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        val_accuracy += acc.item() / len(train_loader)
        val_loss += loss.item() / len(train_loader)

    return epoch_loss, epoch_accuracy, val_loss, val_accuracy


def main():
    seed_everything()
    model = get_vit_model()
    train_loader, val_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)

    run_epoch(model, train_loader, val_loader, criterion, optimizer)


if __name__ == "__main__":
    main()
