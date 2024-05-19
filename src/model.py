import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy


class BoWClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class LitBoWClassifier(LightningModule):
    def __init__(self, input_dim, output_dim, lr=1e-3):
        super().__init__()
        self.model = BoWClassifier(input_dim, output_dim)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=output_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target = batch['target']

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch['x']
        target = batch['target']

        output = self(x)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # simple LSTM + CNN1D model

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        conv_out = self.conv1(lstm_out)
        conv_out = conv_out.permute(0, 2, 1)
        output = self.fc(conv_out)
        return output


class LitLSTMClassifier(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()
        self.model = LSTMClassifier(input_dim, hidden_dim, output_dim)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=output_dim)

    def forward(self, input_ids):
        # pass variable length input through embedding layer
        packed_input = nn.utils.rnn.pack_padded_sequence(input_ids, input_ids.shape[1], batch_first=True)
        packed_output, (hidden, cell) = self.model.lstm(packed_input)

        # get the last hidden state
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        lstm_out = lstm_out.permute(0, 2, 1)
        conv_out = self.model.conv1(lstm_out)
        conv_out = conv_out.permute(0, 2, 1)
        output = self.model.fc(conv_out)

        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target = batch['target']

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target = batch['target']

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class PhraseClassifier(nn.Module):
    def __init__(self, bert, output_dim):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']

        # Freeze the BERT model parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.sequential = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state
        return self.sequential(hidden_state[:, 0, :])


class LitPhraseClassifier(LightningModule):
    def __init__(self, bert, lr=1e-3):
        super().__init__()
        self.model = PhraseClassifier(bert, output_dim=3)
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
