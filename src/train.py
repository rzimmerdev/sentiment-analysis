import json
import os

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from transformers import BertModel

from model import LitPhraseClassifier, LitBoWClassifier, LitLSTMClassifier
from dataset import FinancialPhraseDataset


def train(model_name=None, model_parameters=None, model_path=None, max_epochs=5):
    dataset = FinancialPhraseDataset()
    train_loader, val_loader = dataset.get_data_loaders(batch_size=8, num_workers=4, train_size=0.9)

    model = load_model(model_name=model_name
                       model_parameters=model_parameters)

    if model_path is not None:
        model_path = model_path + '.ckpt'
        try:
            model = model.load_from_checkpoint(model_path, bert=bert)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training from scratch...")
    else:
        model_path = 'lit-phrases-bert' + '.ckpt'

    log_dir = os.path.join('logs/sentiment-analysis', 'version_0')
    logger = CSVLogger("logs", name="sentiment-analysis", version=len(os.listdir(log_dir)))
    trainer = Trainer(max_epochs=max_epochs, logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # save model
    trainer.save_checkpoint(model_path)


def load_model(model_name, model_parameters, pre_trained):
    if model_name == 'LitBoWClassifier':
        model = LitBoWClassifier(**model_parameters)
    elif model_name == 'LitLSTMClassifier':
        model = LitLSTMClassifier(**model_parameters)
    elif model_name == 'LitPhraseClassifier':
        bert = BertModel.from_pretrained(args.pre_trained)
        model = LitPhraseClassifier(bert=bert, **model_parameters)
    else:
        raise ValueError("Unknown model type")

    return model


if __name__ == '__main__':
    # read optinos from command line
    import argparse

    parser = argparse.ArgumentParser(description='Train a sentiment analysis model')
    parser.add_argument('--model', type=str, help='Model architecture', default='LitPhraseClassifier')
    parser.add_argument('--model-parameters', type=json.loads, help='Model hyperparameters', default='{}')
    parser.add_argument('--pre-trained', type=str, help='Path to a pre-trained model', default=None)
    parser.add_argument('--max-epochs', type=int, help='Maximum number of epochs', default=5)
    args = parser.parse_args()
    train(model_name=args.model,
          model_parameters=args.model_parameters,
          model_path=args.pre_trained,
          max_epochs=args.max_epochs)
