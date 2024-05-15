# https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
# https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/download?datasetVersionNumber=5
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PhraseDataset(Dataset):
    def __init__(self, x_col, y_col, tokenizer: torch.nn.Module, data: pd.DataFrame, max_len: int):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

        self.x_col = x_col
        self.y_col = y_col

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

        return {
            "phrase": phrase,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


class FinancialPhraseDataset(PhraseDataset):
    def __init__(self,
                 tokenizer: torch.nn.Module = None,
                 path='data/all-data.csv',
                 max_len=512):
        # if file doesnt exist, test if archive.zip (downloaded from kaggle) exists
        try:
            self.data = pd.read_csv(path, header=None, encoding='latin-1')
        except FileNotFoundError:
            try:
                import zipfile
                with zipfile.ZipFile('data/archive.zip', 'r') as zip_ref:
                    zip_ref.extractall('data')
                self.data = pd.read_csv(path, header=None, encoding='latin-1')
            except FileNotFoundError:
                raise FileNotFoundError("File not found. Please download the dataset from Kaggle.")

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        super().__init__(x_col=1, y_col=0, tokenizer=tokenizer, data=self.data, max_len=max_len)

    def preprocess(self):
        self.data[0] = self.data[0].map({'neutral': 0, 'positive': 1, 'negative': 2})
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)
        return self.data


if __name__ == '__main__':
    dataset = FinancialPhraseDataset()
    print(dataset[0])
