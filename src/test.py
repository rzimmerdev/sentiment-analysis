import torch
from tqdm import tqdm
from transformers import BertModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset import FinancialPhraseDataset
from src.model import LitPhraseClassifier


def test(model_path):
    dataset = FinancialPhraseDataset()
    test_loader = dataset.get_data_loaders(batch_size=8, num_workers=4, train_size=0.9, train=False)

    bert = BertModel.from_pretrained('bert-base-uncased')

    lit_model = LitPhraseClassifier.load_from_checkpoint(model_path, bert=bert)
    lit_model.eval()

    y_true = []
    y_pred = []

    for batch in tqdm(test_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['target']

        output = lit_model(input_ids, attention_mask)
        _, predicted = torch.max(output, dim=1)

        y_true.extend(target.tolist())
        y_pred.extend(predicted.tolist())

    evaluate(y_pred, y_true)


def evaluate(y_pred, y_true):
    mapping = {1: 'positive', 0: 'neutral', 2: 'negative'}

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    unique_labels = set(y_true)

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true))
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {mapping[label]}: {accuracy:.3f}')

    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)

    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)


# Example usage
if __name__ == "__main__":
    test('lit-phrases-bert.ckpt')
