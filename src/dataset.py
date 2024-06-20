import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PhraseDataset(Dataset):
    def __init__(self, x_col, y_col, tokenizer: torch.nn.Module, data: pd.DataFrame, max_len: int):
        """
        Initializes the dataset

        Parameters
        ----------
        x_col: str
            The column name of the input data
        y_col: str
            The column name of the target data
        tokenizer: torch.nn.Module
            The tokenizer to use
        data: pd.DataFrame
            The data to use
        max_len: int
            The maximum length of the input data
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

        self.x_col = x_col
        self.y_col = y_col

        self.num_classes = len(self.data[self.y_col].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        phrase = str(self.data.loc[item, self.x_col])
        encoding = self.tokenizer.encode_plus(
            phrase,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True
        )

        label = self.data.loc[item, self.y_col]
        target = F.one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes)

        return {
            "phrase": phrase,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "target": target.float()
        }


class Splitter:
    def __init__(self, train_size=0.8, seed=42, folder='data'):
        """"
        Initializes the splitter

        Parameters
        ----------
        train_size: float
            The proportion of the data to use for training
        seed: int
            The seed to use
        folder: str
            The folder to save the data
        """
        self.train_size = train_size
        self.seed = seed
        self.folder = folder

    @property
    def paths(self):
        """"Returns the paths of the train and test data."""
        folder_path = self.folder.split('/')
        if len(folder_path) > 1:
            folder = "/".join(folder_path[:-1])
        else:
            folder = folder_path[0]

        train_path = f'{folder}/train.csv'
        test_path = f'{folder}/test.csv'

        return train_path, test_path

    def split(self, df: pd.DataFrame):
        """
        Splits the data into train and test sets

        Parameters
        ----------
        df: pd.DataFrame
            The data to split
        """
        indices = torch.randperm(len(df)).tolist()

        train_size = int(self.train_size * len(df))

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_df = df.loc[train_indices]
        test_df = df.loc[test_indices]

        train_path, test_path = self.paths

        train_df.to_csv(train_path, index=False, header=False)
        test_df.to_csv(test_path, index=False, header=False)

        return train_path, test_path


class FinancialPhraseDataset(PhraseDataset):
    """
    __getitem__ returns a dictionary with the following keys:
    - phrase: the original phrase
    - input_ids: the tokenized phrase
    - attention_mask: the attention mask
    - target: the target sentiment

    """

    def __init__(self,
                 tokenizer: torch.nn.Module = None,
                 path='data/archive.zip',
                 filename='data/all-data.csv',
                 max_len=512,
                 seed=None):
        """
        Initializes the dataset

        Parameters
        ----------
        tokenizer: torch.nn.Module
            The tokenizer to use
        path: str
            The path to the dataset
        filename: str
            The filename of the dataset
        max_len: int
            The maximum length of the input data
        seed: int
            The seed to use
        """
        splitter = Splitter(seed=seed)

        train_path, test_path = splitter.paths

        try:
            self.data = pd.read_csv(train_path, header=None, encoding='latin-1')
            self.test_data = pd.read_csv(test_path, header=None, encoding='latin-1')
        except FileNotFoundError:
            try:
                import zipfile
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall('data')
                self.data = pd.read_csv(filename, header=None, encoding='latin-1')
            except FileNotFoundError:
                raise FileNotFoundError("File not found. Please download the dataset from Kaggle.")
            splitter.split(self.data)

            self.data = pd.read_csv(train_path, header=None, encoding='latin-1')
            self.test_data = pd.read_csv(test_path, header=None, encoding='latin-1')

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        super().__init__(x_col=1, y_col=0, tokenizer=tokenizer, data=self.data, max_len=max_len)
        self.data = self.preprocess(self.data)
        self.test_data = self.preprocess(self.test_data)
        self.seed = seed

        self.data = self.balance_classes()
        self.data = self.data.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def balance_classes(self, oversample=False):
        class_counts = self.data[0].value_counts()

        new_data = pd.DataFrame(columns=self.data.columns)

        if oversample:
            # oversample the minority classes
            max_class_count = class_counts.max()
            for class_label, count in class_counts.items():
                class_data = self.data[self.data[0] == class_label]
                oversampled_data = class_data.sample(n=max_class_count - count, replace=True, random_state=self.seed)
                new_data = pd.concat([new_data, oversampled_data])

        else:
            # undersample the majority classes
            min_class_count = class_counts.min()
            for class_label, count in class_counts.items():
                class_data = self.data[self.data[0] == class_label]
                undersampled_data = class_data.sample(n=min_class_count, random_state=self.seed)
                new_data = pd.concat([new_data, undersampled_data])

        return new_data

    def preprocess(self, data):
        """
        Preprocesses the data by converting the target labels to integers

        Parameters
        ----------
        data: pd.DataFrame
            The data to preprocess
        """
        data[0] = data[0].map({'neutral': 0, 'positive': 1, 'negative': 2})
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data

    def get_data_loaders(self, batch_size=8, shuffle=True, num_workers=0, train_size=0.8, train=True):
        """
        Returns the train and validation data loaders

        Parameters
        ----------
        batch_size: int
            The batch size
        shuffle: bool
            Whether to shuffle the data
        num_workers: int
            The number of workers to use for loading the data
        train_size: float
            The proportion of the data to use for training
        train: bool
            Whether to return the train and validation data loaders or the test data loader
        """
        if train:
            train_size = int(train_size * len(self.data))
            val_size = len(self.data) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(self, [train_size, val_size])

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

            return train_loader, val_loader
        else:
            test_dataset = PhraseDataset(x_col=1,
                                         y_col=0,
                                         tokenizer=self.tokenizer,
                                         data=self.test_data,
                                         max_len=self.max_len)

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

            return test_loader


if __name__ == '__main__':
    dataset = FinancialPhraseDataset()
    train_loader, _ = dataset.get_data_loaders(batch_size=8, num_workers=4, train_size=0.9)

    for batch in train_loader:
        print(batch)
        break
