import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence


class Word2VecVectorizer:
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.model = None

    def fit(self, sentences):
        self.model = Word2Vec(sentences, vector_size=self.embedding_dim, window=5, min_count=1, workers=4)

    def transform(self, sentences):
        embeddings = [torch.tensor(np.array([self.model.wv[word] for word in sentence if word in self.model.wv]), dtype=torch.float32)
                      for sentence in sentences]
        return pad_sequence(embeddings, batch_first=True, padding_value=0.0)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = Word2Vec.load(path)


class Word2VecClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


class LitWord2VecClassifier(LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim=3, lr=1e-3, num_layers=1):
        super().__init__()
        self.model = Word2VecClassifier(embedding_dim, hidden_dim, output_dim, num_layers)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=output_dim)
        self.vectorizer = Word2VecVectorizer(embedding_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to Word2Vec embeddings
        input_ids = self.vectorizer.transform(texts).to(self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to Word2Vec embeddings
        input_ids = self.vectorizer.transform(texts).to(self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to Word2Vec embeddings
        input_ids = self.vectorizer.transform(texts).to(self.device)

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
        tokenized_texts = [text.split() for text in train_texts]
        self.vectorizer.fit(tokenized_texts)
