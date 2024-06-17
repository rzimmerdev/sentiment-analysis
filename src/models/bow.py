import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from sklearn.feature_extraction.text import CountVectorizer


class BoWVectorizer:
    """A Bag of Words vectorizer."""
    def __init__(self, max_features=4000):
        """
        Initializes the vectorizer.
        
        Parameters
        ----------
        max_features: int
            The maximum number of features to use
        """
        self.vectorizer = CountVectorizer(max_features=int(max_features))

    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform the texts.
        """
        return self.vectorizer.fit_transform(texts).toarray()

    def transform(self, texts):
        """
        Transform the texts.

        Parameters
        ----------
        texts: list
            The texts to transform

        Returns
        -------
        np.ndarray
            The transformed texts
        """
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
    """A Bag of Words classifier."""
    def __init__(self, input_dim, hidden_layers=5, output_dim=3):
        """
        Initializes the classifier.

        Parameters
        ----------
        input_dim: int
            The input dimension
        hidden_layers: int
            The number of hidden layers
        output_dim: int
            The output dimension
        """
        super().__init__()

        self.softmax = nn.Softmax(dim=1)

        self.sequential = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU()
            ) for _ in range(hidden_layers - 1)],
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim),
            self.softmax
        )

    def forward(self, x):
        return self.sequential(x)


class LitBowClassifier(LightningModule):
    def __init__(self, input_dim, hidden_layers=5, lr=1e-3):
        """
        Initializes the LightningModule.

        Parameters
        ----------
        input_dim: int
            The input dimension
        hidden_layers: int
            The number of hidden layers
        lr: float
            The learning rate
        """
        super().__init__()
        input_dim = int(input_dim)
        self.model = BowClassifier(input_dim, hidden_layers=hidden_layers)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=3)
        self.vectorizer = BoWVectorizer(max_features=input_dim)

    def forward(self, x):
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

        # Transform texts to BoW vectors
        input_ids = torch.tensor(self.vectorizer.transform(texts), dtype=torch.float32, device=self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(torch.argmax(output, dim=1), torch.argmax(target, dim=1))

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

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

        # Transform texts to BoW vectors
        input_ids = torch.tensor(self.vectorizer.transform(texts), dtype=torch.float32, device=self.device)

        output = self(input_ids)

        loss = self.loss(output, target)
        acc = self.accuracy(output, target)

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
        self.vectorizer.fit_transform(train_texts)
