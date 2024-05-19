from lightning import Trainer
from transformers import BertModel

from src.dataset import FinancialPhraseDataset
from src.model import LitPhraseClassifier


def test(model_path):
    dataset = FinancialPhraseDataset()
    test_loader = dataset.get_data_loaders(batch_size=8, num_workers=4, train_size=0.9, train=False)

    bert = BertModel.from_pretrained('bert-base-uncased')

    # Load the pre-trained model
    lit_model = LitPhraseClassifier.load_from_checkpoint(model_path, bert=bert)

    # Ensure the model is in evaluation mode
    lit_model.eval()

    tester = Trainer()
    results = tester.test(lit_model, dataloaders=test_loader)

    print(f"Test results: {results}")
    print("Testing complete.")


# Example usage
if __name__ == "__main__":
    # Path to the directory containing the saved model
    # Run the test method
    test('lit-phrases-bert.ckpt')
