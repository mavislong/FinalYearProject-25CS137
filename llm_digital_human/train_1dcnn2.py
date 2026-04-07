import os
import glob
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from scipy import stats  

DATA_ROOT = "npz_data"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AudioAUS_Dataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(glob.glob(os.path.join(root, "*.npz")))
        if len(self.paths) ==0:
            raise RuntimeError(f"No npz file")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        audio =data["audio"].astype(np.float32)  
        lip =data["lips"].astype(np.float32)    
        audio= torch.from_numpy(audio).transpose(0, 1)
        lip =torch.from_numpy(lip).transpose(0, 1)    
        return audio,lip


def coll_fn(batch):
    
    audios, lip = zip(*batch)

    max_T = max(a.shape[1] for a in audios)
    padded_audios = []
    padded_lips = []

    for a, l in zip(audios, lips):
        T = a.shape[1]
        if T < max_T:
            pad_T = max_T - T
            last_a = a[:, -1:].repeat(1, pad_T)
            last_l = l[:, -1:].repeat(1, pad_T)
            a = torch.cat([a, last_a], dim=1)
            l = torch.cat([l, last_l], dim=1)
        padded_audios.append(a)
        padded_lips.append(l)

    audios = torch.stack(padded_audios, dim=0)  
    lips = torch.stack(padded_lips, dim=0)      
    return audios, lips




class CNN1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, out_ch, kernel_size=1)  
        )

    def forward(self, x):
        return self.net(x)  



def train_one_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = 0.0
    for audio, lips in tqdm(loader, desc="train", leave=False):
        audio = audio.to(DEVICE)
        lips = lips.to(DEVICE)
        opt.zero_grad()
        pred = model(audio)              
        loss = criterion(pred, lips)     
        loss.backward()
        opt.step()

        total_loss += loss.item() * audio.size(0)
    #print(total_loss / len(loader.dataset))
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for audio, lips in loader:
            audio = audio.to(DEVICE)
            lips = lips.to(DEVICE)
            pred = model(audio)
            loss = criterion(pred, lips)
            total_loss += loss.item() * audio.size(0)
    return total_loss / len(loader.dataset)




def per_au_correlation(model, loader, out_ch):
    
    model.eval()
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for audio, lips in loader:
            audio = audio.to(DEVICE)
            lips = lips.to(DEVICE)
            pred = model(audio)  
            pred_np = pred.cpu().numpy()
            lips_np = lips.cpu().numpy()  
            all_pred.append(pred_np)
            all_gt.append(lips_np)

    if not all_pred:
        return [0.0] * out_ch
    all_pred = np.concatenate(all_pred, axis=0)  
    all_gt = np.concatenate(all_gt, axis=0)      
    N, D, T = all_pred.shape
    assert D == out_ch

    corr_aus = []
    for d in range(D):
        x = all_pred[:,d,:].reshape(-1)
        y = all_gt[:, d,:].reshape(-1)
        if x.std() < 1e-6 or y.std() < 1e-6:
            corr_aus.append(0.0)
        else:
            corr, _ = stats.pearsonr(x, y)
            corr_aus.append(float(corr))
    return corr_aus

'''
def inf_time(model,loader,num_batches=10):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (audio, lips) in enumerate(loader):
            if i >= num_batches:
                break
            audio = audio.to(DEVICE)
            start = time.perf_counter()
            _ = model(audio)
            end = time.perf_counter()
            times.append(end - start)

    if not times:
        print("There is no batches for time.")
        return

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(
        f"Inference time over {len(times)} batches: "
        f"avg {avg_time*1000:.2f} ms, std {std_time*1000:.2f} ms"
    )
'''


def main():
    dataset = AudioAUS_Dataset(DATA_ROOT)
    n_total = len(dataset)
    n_test = int(n_total * TEST_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=coll_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=coll_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=coll_fn
    )

    
    sample_audio,sample_lips = dataset[0]
    in_ch = sample_audio.shape[0]  
    out_ch = sample_lips.shape[0]  

    model = CNN1D(in_ch, out_ch).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ''' 
    print(n_total)
    print(n_train, n_val,n_test)
    print(in_ch,out_ch)
    print(DEVICE)
    '''
    best_val = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "best_1dcnn.pt")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "in_ch": in_ch,
                    "out_ch": out_ch,
                },
                ckpt_path,
            )
            print(f"Saved best model to {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_mse = eval_epoch(model, test_loader, criterion)
    print()
    print( "FINAL TEST RESULTS")
    print(f"Test MSE: {test_mse:.4f}")

    
    per_au_corr = per_au_correlation(model, test_loader, out_ch)
    print("Per‑AU Pearson correlations on test set:")
    for i, c in enumerate(per_au_corr):
        print(f"  AU dim {i}: corr = {c:.3f}")

    
    #print("\nTime for inference")
    #inf_time(model, test_loader, num_batches=10)


if __name__ == "__main__":
    main()
