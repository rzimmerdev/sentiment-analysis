import os
from enum import Enum

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from models.transformer import LitTransformerClassifier
from models.bow import LitBowClassifier
from models.w2v import LitW2VClassifier
from dataset import FinancialPhraseDataset
from plot import plot_training_history


def train(model, max_epochs=5, batch_size=64, num_workers=4, lr=1e-5, path='checkpoints'):
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
    path: str
        The path to save the model and load previous models from
    """
    dataset = FinancialPhraseDataset()
    print(f"Training {model} model")
    print(f"Maimum number of epochs: {max_epochs}. "
          f"Batch size: {batch_size}. Number of workers: {num_workers}. Learning rate: {lr}")
    train_loader, val_loader = dataset.get_data_loaders(batch_size=batch_size, num_workers=num_workers, train_size=0.8)
    lr = float(lr)

    if not os.path.exists(path):
        os.makedirs(path)

    if model == 'bow':
        model_path = f'{path}/bow'
        lit_model = LitBowClassifier(2000, 6, lr=lr)
        if not os.path.exists(f'{path}/bow.pkl'):
            lit_model.fit_vectorizer(dataset.data[1])
        else:
            lit_model.load(model_path + '.pth', model_path + '.pkl')
    elif model == 'w2v':
        model_path = f'{path}/w2v'
        lit_model = LitW2VClassifier(hidden_size=1024, hidden_layers=6, lr=lr)
        if not os.path.exists(f'{path}/w2v.model'):
            pass
        else:
            lit_model.load(model_path + '.pth')
    elif model == 'transformer':
        model_path = f'{path}/transformer'
        lit_model = LitTransformerClassifier(hidden_layers=3, lr=lr)
        if not os.path.exists(f'{path}/transformer.ckpt'):
            pass
        else:
            lit_model = LitTransformerClassifier.load_from_checkpoint(model_path + '.ckpt', hidden_layers=3, lr=lr)
    else:
        raise ValueError('Unknown model')

    log_dir = os.path.join(f'logs/{model}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min',
        min_delta=1e-4
    )  # if val_loss does not improve for 7 epochs, stop training

    logger = CSVLogger("logs", name=model)
    trainer = Trainer(max_epochs=max_epochs, logger=logger, callbacks=[early_stop_callback], log_every_n_steps=1)
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # save model
    if model == 'transformer':
        trainer.save_checkpoint(model_path + '.ckpt')
    elif model == 'bow':
        lit_model.save(model_path + '.pth', model_path + '.pkl')
    elif model == 'w2v':
        lit_model.save(model_path + '.pth')

    # save results
    # list all versions, and access metrics from latest
    version = max([int(version.split('_')[-1]) for version in os.listdir(f'logs/{model}')])
    log_dir = os.path.join(f'logs/{model}', f'version_{version}')
    plot_training_history(log_dir, model)

    # save train params to txt
    with open(f'{model_path}.txt', 'w') as f:
        f.write(f"Training **{model}** model\n")
        f.write(f"epochs, batch_size, num_workers, lr\n")
        f.write(f"{max_epochs}, {batch_size}, {num_workers}, {lr}\n")


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
