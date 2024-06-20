from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score, log_loss, classification_report
from models.transformer import LitTransformerClassifier
from models.bow import LitBowClassifier
from models.w2v import LitW2VClassifier
from dataset import FinancialPhraseDataset


def evaluate(model, model_dir=None):
    """"
    Evaluate a sentiment analysis model.

    The metrics calculated are:
    - Test Accuracy
    - Test Loss
    - F1 Score
    - Confusion Matrix
    - ROC Curve and AUC
    - AIC Score

    Parameters
    ----------
    model: str
        The model to evaluate, one of 'transformer', 'bow', 'w2v'.
    model_dir: str
        The path to the model to evaluate
    """
    dataset = FinancialPhraseDataset()
    test_loader = dataset.get_data_loaders(train=False, batch_size=64, num_workers=8, train_size=1)
    model_path = f"{model_dir}/{model}"

    if model == 'bow':
        lit_model = LitBowClassifier(2000, 6)
        weights = model_path + '.pth'
        vectorizer = model_path + '.pkl'
        lit_model.load(weights, vectorizer)
    elif model == 'w2v':
        lit_model = LitW2VClassifier(hidden_size=1024, hidden_layers=6)
        weights = model_path + '.pth'
        lit_model.load(weights)
    elif model == 'transformer':
        weights = model_path + '.ckpt'
        lit_model = LitTransformerClassifier.load_from_checkpoint(weights, hidden_layers=3)
    else:
        raise ValueError('Unknown model')

    lit_model.eval()

    # Calculate metrics
    all_outputs = []
    all_labels = []

    for idx, batch in enumerate(test_loader):
        output, target, _, _ = lit_model.test_step(batch, idx)

        all_outputs.extend(output.tolist())
        all_labels.extend(target.tolist())

    outputs = np.array(all_outputs)
    labels = np.array(all_labels)

    pred = np.argmax(outputs, axis=1)
    target = np.argmax(labels, axis=1)

    # Accuracy (argmax, argmax
    test_accuracy = accuracy_score(target, pred)

    # Confusion Matrix
    conf_matrix = confusion_matrix(target, pred)

    # F1 Score
    f1 = f1_score(target, pred, average='weighted')

    # Log loss
    test_loss = log_loss(all_labels, all_outputs)

    # ROC Curve and AUC
    for i in range(3):
        fpr, tpr, _ = roc_curve(labels[:, i], outputs[:, i])
        roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'results/{model}/class_{i}_roc.png')

    plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap='viridis')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'results/{model}/confusion_matrix.png')

    # AIC Score (assuming a binomial model, and AIC = 2k - 2ln(L))
    # k = num of model params
    k = sum(p.numel() for p in lit_model.parameters())
    L = -test_loss * len(all_labels)  # log-likelihood
    AIC = 2 * k - 2 * L

    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Loss: {test_loss}')
    print(f'AIC: {AIC}')
    print(conf_matrix)

    # micro and macro scores for: precision, recall, f1
    report = classification_report(target, pred, target_names=['neutral', 'positive', 'negative'])
    print(report)
    # save to csv
    with open(f'results/{model}/report.txt', 'w') as f:
        f.write(report)


class Models(Enum):
    transformer = 'transformer'
    bow = 'bow'
    w2v = 'w2v'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a sentiment analysis model')
    parser.add_argument('--model', type=Models, help='Model to evaluate', default=Models.transformer)
    parser.add_argument('--pre-trained', type=str, help='Path to a pre-trained model', default=None)
    args = parser.parse_args()
    evaluate(args.model.value, args.pre_trained)
