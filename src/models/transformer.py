import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from transformers import BertModel


class TransformerClassifier(nn.Module):
    def __init__(self, bert, hidden_layers=5, output_dim=3):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']

        # Freeze the BERT model parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # unfreeze last
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        self.sequential = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU()
            ) for _ in range(hidden_layers - 1)],
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state
        return self.sequential(hidden_state[:, 0, :])


class LitTransformerClassifier(LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.model = TransformerClassifier(bert, output_dim=3)
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
        acc = self.accuracy(output, target)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['target']

        output = self(input_ids, attention_mask)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['target']

        output = self(input_ids, attention_mask)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def save(self, path):
        torch.save(self.state_dict(), path)
