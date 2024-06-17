import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Word2VecVectorizer:
    """
    A Word2Vec vectorizer.
    """
    def __init__(self, embedding_dim=300):
        """
        Initializes the vectorizer.

        Parameters
        ----------
        embedding_dim: int
            The dimension of the word embeddings
        """
        self.embedding_dim = embedding_dim
        self.model = None

    def fit(self, sentences):
        self.model = Word2Vec(sentences, vector_size=self.embedding_dim, window=5, min_count=1, workers=4)

    def transform(self, sentences):
        """
        Transform the sentences to embeddings.

        Parameters
        ----------
        sentences: list of list of str
            The sentences to transform

        Returns
        -------
        torch.Tensor
            The transformed embeddings
        """
        embeddings = []
        lengths = []

        for sentence in sentences:
            vectors = [self.model.wv[word] for word in sentence if word in self.model.wv]
            embeddings.append(torch.tensor(np.array(vectors), dtype=torch.float32))
            lengths.append(len(vectors))

        padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(np.array(lengths), dtype=torch.long)

        return padded_embeddings, lengths

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = Word2Vec.load(path)


class Word2VecClassifier(nn.Module):
    """
    A Word2Vec-based classifier.
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers=1):
        """
        Initializes the classifier.

        Parameters
        ----------
        embedding_dim: int
            The dimension of the word embeddings
        hidden_dim: int
            The dimension of the hidden layer
        output_dim: int
            The output dimension
        num_layers: int
            The number of LSTM layers
        """
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        hidden = hidden[-1]
        logits = self.classifier(hidden)
        return logits


class LitWord2VecClassifier(LightningModule):
    """
    A LightningModule for training the Word2Vec classifier.
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim=3, lr=1e-3, num_layers=1):
        """
        Initializes the LightningModule.

        Parameters
        ----------
        embedding_dim: int
            The dimension of the word embeddings
        hidden_dim: int
            The dimension of the hidden layer
        output_dim: int
            The output dimension
        lr: float
            The learning rate
        num_layers: int
            The number of LSTM layers
        """
        super().__init__()
        self.model = Word2VecClassifier(embedding_dim, hidden_dim, output_dim, num_layers)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=output_dim)
        self.vectorizer = Word2VecVectorizer(embedding_dim)

    def forward(self, x, lengths):
        return self.model(x, lengths)

    def training_step(self, batch, batch_idx):
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to Word2Vec embeddings
        input_ids, lengths = self.vectorizer.transform([text.split() for text in texts])
        input_ids, target = input_ids.to(self.device), target.to(self.device)

        output = self(input_ids, lengths)

        loss = self.loss(output, target)
        acc = self.accuracy(torch.argmax(output, dim=1), torch.argmax(target, dim=1))

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        texts = batch['phrase']
        target = batch['target']

        # Transform texts to Word2Vec embeddings
        input_ids, lengths = self.vectorizer.transform([text.split() for text in texts])
        input_ids, target = input_ids.to(self.device), target.to(self.device)

        output = self(input_ids, lengths)

        loss = self.loss(output, torch.argmax(target, dim=1))
        acc = self.accuracy(output, torch.argmax(target, dim=1))

        return output, target, loss, acc

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
