import os
from enum import Enum

import pandas as pd
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
from models.transformer import LitTransformerClassifier
from models.bow import LitBowClassifier
from models.w2v import LitWord2VecClassifier
from dataset import FinancialPhraseDataset


def plot_training_history(log_dir, model):
    metrics = pd.read_csv(f'{log_dir}/metrics.csv')
    plt.figure()

    # make a 1/10 moving average
    window = 15
    metrics['train_loss_step'] = metrics['train_loss_step'].rolling(window=window).mean()
    metrics['train_acc_step'] = metrics['train_acc_step'].rolling(window=window).mean()

    plt.plot(metrics['step'], metrics['train_loss_step'].bfill(), label='Training Loss')
    plt.plot(metrics['step'], metrics['train_acc_step'].bfill(), label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Training History - {model}')
    plt.legend()
    plt.savefig(f'{model}_training_history.png')


def train(model, model_path=None, max_epochs=5):
    dataset = FinancialPhraseDataset()
    train_loader, _ = dataset.get_data_loaders(batch_size=8, num_workers=4, train_size=1)

    if model == 'transformer':
        if model_path is not None:
            model_path = model_path + '.ckpt'
            lit_model = LitTransformerClassifier.load_from_checkpoint(model_path)
        else:
            model_path = 'checkpoints/transformer'
            lit_model = LitTransformerClassifier()
    elif model == 'bow':
        if model_path is not None:
            lit_model = LitBowClassifier(input_dim=1000)
            model_path = model_path + '.pth'
            vectorizer_path = model_path + '.pkl'
            lit_model.load(model_path, vectorizer_path)
        else:
            model_path = 'checkpoints/bow'
            lit_model = LitBowClassifier(input_dim=1000)
            lit_model.fit_vectorizer(dataset.data[1])
    elif model == 'w2v':
        if model_path is not None:
            lit_model = LitWord2VecClassifier(embedding_dim=100, hidden_dim=128)
            model_path = model_path + '.pth'
            vectorizer_path = model_path + '.model'
            lit_model.load(model_path, vectorizer_path)
        else:
            model_path = 'checkpoints/w2v'
            lit_model = LitWord2VecClassifier(embedding_dim=100, hidden_dim=128)
            lit_model.fit_vectorizer(dataset.data[1])
    else:
        raise ValueError('Unknown model')

    log_dir = os.path.join('logs/sentiment-analysis')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = CSVLogger("logs", name="sentiment-analysis")
    early_stopping = EarlyStopping(
        monitor='train_loss_epoch',
        patience=5,
        mode='min',
        verbose=True,
        min_delta=1e-2,
    )
    trainer = Trainer(max_epochs=max_epochs, logger=logger, callbacks=[early_stopping])
    trainer.fit(model=lit_model, train_dataloaders=train_loader)

    # if directory does not exist, create it
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # save model
    if model == 'transformer':
        trainer.save_checkpoint(model_path + '.ckpt')
    elif model == 'bow':
        lit_model.save(model_path + '.pth', model_path + '.pkl')
    elif model == 'w2v':
        lit_model.save(model_path + '.pth', model_path + '.model')

    # save results
    # list all versions, and access metrics from latest
    version = max([int(version.split('_')[-1]) for version in os.listdir('logs/sentiment-analysis')])
    log_dir = os.path.join('logs/sentiment-analysis', f'version_{version}')
    plot_training_history(log_dir, model)


class Models(Enum):
    transformer = 'transformer'
    bow = 'bow'
    w2v = 'w2v'


if __name__ == '__main__':
    # read optinos from command line
    import argparse

    parser = argparse.ArgumentParser(description='Train a sentiment analysis model')
    parser.add_argument('--model', type=Models, help='Model to train', default=Models.transformer)
    parser.add_argument('--pre-trained', type=str, help='Path to a pre-trained model', default=None)
    parser.add_argument('--max-epochs', type=int, help='Maximum number of epochs', default=5)
    args = parser.parse_args()
    train(args.model.value, args.pre_trained, args.max_epochs)
