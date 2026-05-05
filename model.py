import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from torch.utils.data import Dataset, DataLoader
import argparse
import random


# ==================== Dataset ====================

class DepressionDataset(Dataset):

    def __init__(self, mfcc, prosody, au, gender, labels,
                 phase='train', means=None, stds=None):

        self.phase = phase
        self.means = means
        self.stds = stds

        self.audio_features = np.concatenate([mfcc, prosody], axis=2)

        self.audio_features, self.au = self.normalize_all(
            self.audio_features, au)

        self.gender = gender.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return (
            torch.FloatTensor(self.audio_features[idx]),
            torch.FloatTensor(self.au[idx]),
            torch.FloatTensor([self.gender[idx]]),
            torch.FloatTensor([self.labels[idx]])
        )

    def normalize_all(self, audio, au):

        if self.phase == "train":

            self.means = {
                "audio": np.mean(audio, axis=(0, 1)),
                "au": np.mean(au, axis=(0, 1))
            }

            self.stds = {
                "audio": np.std(audio, axis=(0, 1)),
                "au": np.std(au, axis=(0, 1))
            }

        audio = (audio - self.means["audio"]) / (self.stds["audio"] + 1e-8)
        au = (au - self.means["au"]) / (self.stds["au"] + 1e-8)

        return audio, au


# ==================== Model ====================

class BiLSTMBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        x, _ = self.lstm(x)

        return self.norm(x)


class CrossModalAttention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

    def forward(self, audio, au):

        x = torch.cat([audio, au], dim=1)

        return self.proj(x)


class MultiModalDepressionClassifier(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.audio_lstm = BiLSTMBlock(17, config.hidden_dim)
        self.au_lstm = BiLSTMBlock(17, config.hidden_dim)

        self.cross_modal = CrossModalAttention(config.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, audio, au, gender):

        audio_feat = self.audio_lstm(audio).mean(dim=1)
        au_feat = self.au_lstm(au).mean(dim=1)

        fused = self.cross_modal(audio_feat, au_feat)

        return torch.sigmoid(self.classifier(fused))


# ==================== Trainer ====================

class Trainer:

    def __init__(self, model, config):

        self.model = model.to(device)

        self.criterion = nn.BCELoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr
        )

        self.best_auc = 0

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0

        for audio, au, gender, labels in loader:

            audio, au, gender, labels = \
                [x.to(device) for x in (audio, au, gender, labels)]

            self.optimizer.zero_grad()

            outputs = self.model(audio, au, gender)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):

        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for audio, au, gender, labels in loader:

                audio, au, gender, labels = \
                    [x.to(device) for x in (audio, au, gender, labels)]

                outputs = self.model(audio, au, gender)

                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        metrics = {
            "auc": roc_auc_score(all_labels, all_preds),
            "f1": f1_score(all_labels, (all_preds > 0.5).astype(int)),
            "acc": accuracy_score(all_labels, (all_preds > 0.5).astype(int)),
            "recall": recall_score(all_labels, (all_preds > 0.5).astype(int))
        }

        return metrics

    def save_best_model(self, metrics, path):

        if metrics["auc"] > self.best_auc:

            self.best_auc = metrics["auc"]

            torch.save(self.model.state_dict(), path)

            print(f"Best model saved AUC={self.best_auc:.4f}")


# ==================== Load Data ====================

def load_data():

    with open(f"f:/0000项目文件/项目workspace/0final/data_dict_125.pkl", "rb") as f:
        loaded = pickle.load(f)

    with open(f"f:/0000项目文件/项目workspace/0final/mfcc_data_dict_125.pkl", "rb") as f:
        mfcc = pickle.load(f)

    with open(f"f:/0000项目文件/项目workspace/0final/au_data_dict_125.pkl", "rb") as f:
        au = pickle.load(f)

    with open(f"f:/0000项目文件/项目workspace/0final/audio_data_dict_125.pkl", "rb") as f:
        prosody = pickle.load(f)

    gender = np.array(loaded["_sexs"]).reshape(-1, 1)
    labels = np.array(loaded["_labels"])

    return mfcc, prosody, au, gender, labels


# ==================== Main ====================

if __name__ == "__main__":

    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mfcc, prosody, au, gender, labels = load_data()
    
    mfcc = np.array(mfcc)
    prosody = np.array(prosody)
    au = np.array(au)
    gender = np.array(gender)
    labels = np.array(labels)

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    train_set = DepressionDataset(
        mfcc[train_idx],
        prosody[train_idx],
        au[train_idx],
        gender[train_idx],
        labels[train_idx],
        phase="train"
    )

    val_set = DepressionDataset(
        mfcc[val_idx],
        prosody[val_idx],
        au[val_idx],
        gender[val_idx],
        labels[val_idx],
        phase="val",
        means=train_set.means,
        stds=train_set.stds
    )

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)

    config = parser.parse_args()

    model = MultiModalDepressionClassifier(config)

    trainer = Trainer(model, config)

    for epoch in range(config.epochs):

        train_loss = trainer.train_epoch(train_loader)

        metrics = trainer.validate(val_loader)

        trainer.save_best_model(metrics, "best_model.pth")

        print(
            f"Epoch {epoch+1} | "
            f"Loss {train_loss:.4f} | "
            f"AUC {metrics['auc']:.4f} | "
            f"F1 {metrics['f1']:.4f}"
        )