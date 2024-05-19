import os

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from transformers import BertModel

from model import LitPhraseClassifier
from dataset import FinancialPhraseDataset


def train(model_path=None, max_epochs=5):
    dataset = FinancialPhraseDataset()
    train_loader, val_loader = dataset.get_data_loaders(batch_size=8, num_workers=4, train_size=0.9)

    bert = BertModel.from_pretrained('bert-base-uncased')

    if model_path is not None:
        model_path = model_path + '.ckpt'
        try:
            lit_model = LitPhraseClassifier.load_from_checkpoint(model_path, bert=bert)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training from scratch...")
            lit_model = LitPhraseClassifier(bert=bert)
    else:
        model_path = 'lit-phrases-bert' + '.ckpt'
        lit_model = LitPhraseClassifier(bert=bert)

    log_dir = os.path.join('logs/sentiment-analysis', 'version_0')
    logger = CSVLogger("logs", name="sentiment-analysis", version=len(os.listdir(log_dir)))
    trainer = Trainer(max_epochs=max_epochs, logger=logger)
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # save model
    trainer.save_checkpoint(model_path)


if __name__ == '__main__':
    # read optinos from command line
    import argparse

    parser = argparse.ArgumentParser(description='Train a sentiment analysis model')
    parser.add_argument('--pre-trained', type=str, help='Path to a pre-trained model', default=None)
    parser.add_argument('--max-epochs', type=int, help='Maximum number of epochs', default=5)
    args = parser.parse_args()
    train(args.pre_trained, args.max_epochs)
