import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from transformers import BertModel


class TransformerClassifier(nn.Module):
    """A Transformer-based classifier."""
    def __init__(self, bert, hidden_layers=5, output_dim=3):
        """
        Initializes the classifier.

        Parameters
        ----------
        bert: BertModel
            The pre-trained BERT model
        hidden_layers: int
            The number of hidden layers
        output_dim: int
            The output dimension
        """
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']

        # Freeze the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False

        self.sequential = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU()
            ) for _ in range(hidden_layers - 1)],
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, output_dim),
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input IDs tensor
        attention_mask: torch.Tensor
            The attention mask tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state
        return self.sequential(hidden_state[:, 0, :])

    def load(self, path):
        # save only the classifier state dict, and ignore bert
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)


class LitTransformerClassifier(LightningModule):
    """
    A LightningModule for training the Transformer classifier.
    """
    def __init__(self, hidden_layers=5, lr=1e-3):
        """
        Initializes the LightningModule.

        Parameters
        ----------
        hidden_layers: int
            The number of hidden layers
        lr: float
            The learning rate
        """
        super().__init__()
        bert = BertModel.from_pretrained('prajjwal1/bert-small')
        self.model = TransformerClassifier(bert, hidden_layers=hidden_layers, output_dim=3)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=3)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['target']

        output = self(input_ids, attention_mask)

        loss = self.loss(output, target)
        acc = self.accuracy(torch.argmax(output, dim=1), torch.argmax(target, dim=1))

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['target']

        output = self(input_ids, attention_mask)

        loss = self.loss(output, target)
        acc = self.accuracy(torch.argmax(output, dim=1), torch.argmax(target, dim=1))

        self.log('val_loss', loss, on_epoch=True, on_step=True)
        self.log('val_acc', acc, on_epoch=True, on_step=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['target']

        output = self(input_ids, attention_mask)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        return output, target, loss, acc

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
