import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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
    if not os.path.exists(f'results/{model}'):
        os.makedirs(f'results/{model}')
    plt.figure()

    # Make
    train_rows = metrics[metrics['train_loss_step'].notnull()]
    val_rows = metrics[metrics['val_loss_step'].notnull()]

    # make val_rows so it has the same length as train_rows (move existing rows so they are equally spaced and the backfill)
    val_rows = val_rows.reindex(train_rows.index, method='ffill')
    val_rows['step'] = train_rows['step']

    plot_train = pd.DataFrame(columns=['train_loss'])
    plot_val = pd.DataFrame(columns=['val_loss', 'val_acc'])

    train_window = len(train_rows) // 10
    val_window = len(val_rows) // 10

    plot_train['train_loss'] = train_rows['train_loss_step'].bfill().rolling(window=train_window).mean()

    plot_val['val_loss'] = val_rows['val_loss_step'].bfill().rolling(window=val_window).mean()
    plot_val['val_acc'] = val_rows['val_acc_step'].bfill().rolling(window=val_window).mean()


    # plot training and val loss on same axis
    # plot val acc on right side axis
    fig, ax1 = plt.subplots()

    ax1.plot(plot_train, label='train_loss', color='blue')
    ax1.plot(plot_val['val_loss'], label='val_loss', color='red')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.legend(loc='upper left')

    def to_percent(y, _):
        return f'{y * 100:.0f}%'

    ax2 = ax1.twinx()
    ax2.plot(plot_val['val_acc'], label='val_acc', color='green')
    ax2.set_ylabel('Accuracy', color='green')
    ax2.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title(f'Training History - {model}')
    plt.savefig(f'results/{model}/training.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training history')

    parser.add_argument('--log_dir', type=str, help='The path to the log directory of the model')
    parser.add_argument('--version', type=str, help='The version of the model')
    parser.add_argument('--model', type=str, help='The model name used in the title of the plot')

    args = parser.parse_args()
    log_dir = f'{args.log_dir}/{args.model}' + '/version_' + args.version

    plot_training_history(log_dir, args.model)
