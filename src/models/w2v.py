import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence


class Word2VecVectorizer:
    """
    A Word2Vec vectorizer.
    """
    def __init__(self, embedding_dim=2000):
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
        """
        Fit the Word2Vec model.

        Parameters
        ----------
        sentences: list of list of str
            The sentences to fit the model on
        """
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
        embeddings = [torch.tensor(np.array([self.model.wv[word] for word in sentence if word in self.model.wv]), dtype=torch.float32)
                      for sentence in sentences]
        return pad_sequence(embeddings, batch_first=True, padding_value=0.0)

    def save(self, path):
        """
        Save the Word2Vec model to a file.

        Parameters
        ----------
        path: str
            The path to save the model to
        """
        self.model.save(path)

    def load(self, path):
        """
        Load the Word2Vec model from a file.

        Parameters
        ----------
        path: str
            The path to load the model from
        """
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
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            self.softmax
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


class LitWord2VecClassifier(LightningModule):
    """
    A LightningModule for training the Word2Vec classifier.
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim=3, lr=1e-3, num_layers=8):
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

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        return self.model(x)

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

        # Transform texts to Word2Vec embeddings
        input_ids = self.vectorizer.transform(texts).to(self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(torch.argmax(output, dim=1), torch.argmax(target, dim=1))

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

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

        # Transform texts to Word2Vec embeddings
        input_ids = self.vectorizer.transform(texts).to(self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(torch.argmax(output, dim=1), torch.argmax(target, dim=1))

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True)

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

        # Transform texts to Word2Vec embeddings
        input_ids = self.vectorizer.transform(texts).to(self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True)

        pred = output.argmax(dim=1)
        self.log('test_pred', pred, on_step=True, on_epoch=True)
        self.log('test_target', target, on_step=True, on_epoch=True)

        return output, target, loss, acc

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer to use
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def save(self, model_path, vectorizer_path):
        """
        Save the model and vectorizer.

        Parameters
        ----------
        model_path: str
            The path to save the model to
        vectorizer_path: str
            The path to save the vectorizer to
        """
        torch.save(self.state_dict(), model_path)
        self.vectorizer.save(vectorizer_path)

    def load(self, model_path, vectorizer_path):
        """
        Load the model and vectorizer from files.

        Parameters
        ----------
        model_path: str
            The path to load the model from
        vectorizer_path: str
            The path to load the vectorizer from
        """
        self.load_state_dict(torch.load(model_path))
        self.vectorizer.load(vectorizer_path)

    def fit_vectorizer(self, train_texts):
        """
        Fit the vectorizer.

        Parameters
        ----------
        train_texts: list of str
            The training texts
        """
        tokenized_texts = [text.split() for text in train_texts]
        self.vectorizer.fit(tokenized_texts)
