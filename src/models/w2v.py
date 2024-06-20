import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from transformers import BertTokenizer, BertModel


# use tokenizer from hugging face


class W2VClassifier(nn.Module):
    """A Word2Vec classifier with an LSTM layer."""

    def __init__(self, hidden_size, hidden_layers=5, output_dim=3, device=None):
        super().__init__()
        # Load pre-trained bert tokenizer and embedding
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding = BertModel.from_pretrained('bert-base-uncased')

        # freeze
        for param in self.embedding.parameters():
            param.requires_grad = False

        embed_dim = self.embedding.config.to_dict()['hidden_size']

        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.sequential = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.BatchNorm1d(hidden_size),
            ) for _ in range(hidden_layers - 1)],
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim),
        )

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        return self.sequential(x)

    def transform(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # Move inputs to the device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return self.embedding(**inputs).last_hidden_state


class LitW2VClassifier(LightningModule):
    def __init__(self, hidden_size=1000, hidden_layers=5, lr=1e-3):
        """
        Initializes the LightningModule.

        Parameters
        ----------
        vocab_size: int
            The vocabulary size
        embed_dim: int
            The embedding dimension
        hidden_dim: int
            The LSTM hidden dimension
        lr: float
            The learning rate
        """
        super().__init__()
        self.model = W2VClassifier(hidden_size, hidden_layers, 3, self.device)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=3)

    def forward(self, x):
        return self.model(x)

    def transform(self, texts):  # returns a tensor, where each row is a BoW vector (1 x vocab_size)
        return self.model.transform(texts)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Parameters
        ----------
        batch: dict
            The batch data
        batch_idx: int
            The batch index

        Returns
        -------
        float
            The loss value for the batch
        """
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to W2V vectors
        embedding = self.transform(texts)
        output = self(embedding)

        loss = self.loss(output, target)
        acc = self.accuracy(torch.argmax(output, dim=1), torch.argmax(target, dim=1))

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to W2V vectors
        embedding = self.transform(texts)
        output = self(embedding)

        loss = self.loss(output, target)
        acc = self.accuracy(torch.argmax(output, dim=1), torch.argmax(target, dim=1))

        self.log('val_loss', loss, on_epoch=True, on_step=True)
        self.log('val_acc', acc, on_epoch=True, on_step=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Parameters
        ----------
        batch: dict
            The batch data
        batch_idx: int
            The batch index

        Returns
        -------
        torch.Tensor
            The output tensor
        torch.Tensor
            The target tensor
        float
            The loss value for the batch
        float
            The accuracy value for the batch
        """
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to W2V vectors
        embedding = self.transform(texts)
        output = self(embedding)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        return output, target, loss, acc

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
