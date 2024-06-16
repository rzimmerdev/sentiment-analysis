from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score, log_loss
from models.transformer import LitTransformerClassifier
from models.bow import LitBowClassifier
from models.w2v import LitWord2VecClassifier
from dataset import FinancialPhraseDataset


def evaluate(model, model_path=None):
    dataset = FinancialPhraseDataset()
    test_loader = dataset.get_data_loaders(train=False, batch_size=8, num_workers=4, train_size=1)

    if model == 'bow':
        lit_model = LitBowClassifier(2000, 6)
        vectorizer_path = model_path + '.pkl'
        model_path = model_path + '.pth'
        lit_model.load(model_path, vectorizer_path)
    elif model == 'w2v':
        lit_model = LitWord2VecClassifier(embedding_dim=512, hidden_dim=256, num_layers=8)
        vectorizer_path = model_path + '.model'
        model_path = model_path + '.pth'
        lit_model.load(model_path, vectorizer_path)
    elif model == 'transformer':
        model_path = model_path + '.ckpt'
        lit_model = LitTransformerClassifier.load_from_checkpoint(model_path)
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
        plt.savefig(f'{model}_roc_curve.png')

        print(f'Class {i} ROC AUC: {roc_auc}')

    # AIC Score (assuming a binomial model, and AIC = 2k - 2ln(L))
    # k = num of model params
    k = sum(p.numel() for p in lit_model.parameters())
    L = -test_loss * len(all_labels)  # log-likelihood
    AIC = 2 * k - 2 * L

    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Loss: {test_loss}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'AIC: {AIC}')


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
