import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from sklearn.feature_extraction.text import CountVectorizer


class BoWVectorizer:
    def __init__(self, max_features=1000):
        self.vectorizer = CountVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)


class BowClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers=5, output_dim=3):
        super().__init__()

        self.sequential = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU()
            ) for _ in range(hidden_layers - 1)],
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.sequential(x)


class LitBowClassifier(LightningModule):
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()
        self.model = BowClassifier(input_dim, output_dim=3)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=3)
        self.vectorizer = BoWVectorizer(max_features=input_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to BoW vectors
        input_ids = torch.tensor(self.vectorizer.transform(texts), dtype=torch.float32, device=self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to BoW vectors
        input_ids = torch.tensor(self.vectorizer.transform(texts), dtype=torch.float32, device=self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to BoW vectors
        input_ids = torch.tensor(self.vectorizer.transform(texts), dtype=torch.float32, device=self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def save(self, model_path, vectorizer_path):
        torch.save(self.state_dict(), model_path)
        self.vectorizer.save(vectorizer_path)

    def load(self, model_path, vectorizer_path):
        self.load_state_dict(torch.load(model_path))
        self.vectorizer.load(vectorizer_path)

    def fit_vectorizer(self, train_texts):
        self.vectorizer.fit_transform(train_texts)

# Example usage (assuming train_texts and train_labels are available):
# lit_model = LitPhraseClassifier(input_dim=1000)
# lit_model.fit_vectorizer(train_texts)
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(lit_model, train_dataloader)
