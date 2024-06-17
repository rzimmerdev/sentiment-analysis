import os
from enum import Enum

import pandas as pd
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
from models.transformer import LitTransformerClassifier
from models.bow import LitBowClassifier
from models.w2v import LitWord2VecClassifier
from dataset import FinancialPhraseDataset


def plot_training_history(log_dir, model):
    """"
    Plot the training history of a model.

    Parameters
    ----------
    log_dir: str
        The path to the log directory of the model
    model: str
        The model name used in the title of the plot.
    """
    metrics = pd.read_csv(f'{log_dir}/metrics.csv')
    plt.figure()

    # Make a 1/30 moving average
    window = 10
    metrics['train_loss_step'] = metrics['train_loss_step'].bfill().rolling(window=window).mean()
    metrics['train_acc_step'] = metrics['train_acc_step'].bfill().rolling(window=window).mean()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(metrics['step'], metrics['train_loss_step'], label='Training Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Training Accuracy', color='tab:orange')
    ax2.plot(metrics['step'], metrics['train_acc_step'], label='Training Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title(f'Training History - {model}')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.savefig(f'results/{model}_training_history.png')


def train(model, max_epochs=5, batch_size=64, num_workers=4, lr=1e-5):
    """
    Train a sentiment analysis model.

    Parameters
    ----------
    model: str
        The model to train, one of 'transformer', 'bow', 'w2v'.
    max_epochs: int
        The maximum number of epochs to train for
    batch_size: int
        The batch size
    num_workers: int
        The number of workers to use
    lr: float
        The learning rate
    """
    dataset = FinancialPhraseDataset()
    print(f"Training {model} model")
    print(f"Maimum number of epochs: {max_epochs}. "
          f"Batch size: {batch_size}. Number of workers: {num_workers}. Learning rate: {lr}")
    train_loader, _ = dataset.get_data_loaders(batch_size=batch_size, num_workers=num_workers, train_size=1)
    lr = float(lr)

    if model == 'bow':
        model_path = 'checkpoints/bow'
        lit_model = LitBowClassifier(2000, 6, lr=lr)
        lit_model.fit_vectorizer(dataset.data[1])
    elif model == 'w2v':
        model_path = 'checkpoints/w2v'
        lit_model = LitWord2VecClassifier(embedding_dim=512, hidden_dim=128, num_layers=12, lr=lr)
        lit_model.fit_vectorizer(dataset.data[1])
    elif model == 'transformer':
        model_path = 'checkpoints/transformer'
        lit_model = LitTransformerClassifier(hidden_layers=3, lr=lr)
    else:
        raise ValueError('Unknown model')

    log_dir = os.path.join(f'logs/{model}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = CSVLogger("logs", name=model)
    trainer = Trainer(max_epochs=max_epochs, logger=logger)
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
    version = max([int(version.split('_')[-1]) for version in os.listdir(f'logs/{model}')])
    log_dir = os.path.join(f'logs/{model}', f'version_{version}')
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
    parser.add_argument('--max-epochs', type=int, help='Maximum number of epochs', default=5)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=64)
    parser.add_argument('--num-workers', type=int, help='Number of workers for data loaders', default=4)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
    args = parser.parse_args()
    train(args.model.value, args.max_epochs, args.batch_size, args.num_workers, args.lr)
