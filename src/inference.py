import torch
import os

from utils import load_config, set_logger, setting_seed
from model import EncoderForClassification
from main import calculate_accuracy
from data import get_dataloader

def test_iter(model, inputs, device):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        loss = outputs['loss']
        accuracy = calculate_accuracy(outputs['logits'], inputs['label'])  
    return loss.item(), accuracy

def main(configs):
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Set seed
    setting_seed(configs.SEED)

    # make directory
    savedir = os.path.join(configs.RESULT.savedir, 'test')
    os.makedirs(savedir, exist_ok=True)

    configs.LOG.file_path = os.path.join('test', configs.EXP_NAME)
    _logger = set_logger(configs.LOG.dir_path, configs.LOG.file_path)

    model_path = os.path.join(configs.RESULT.savedir, configs.MODEL.model_name + '_best.pt')
    model = EncoderForClassification(configs.MODEL)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    test_loader = get_dataloader(configs.DATASET, split='test')

    test_loss = 0
    test_accuracy = 0
    for inputs in test_loader:
        test_loss_iter, test_accuracy_iter = test_iter(model, inputs, device)
        test_loss += test_loss_iter
        test_accuracy += test_accuracy_iter
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    _logger.info(f"[EVAL] loss-{test_loss:.2f}, acc-{test_accuracy*100:.2f}%")


if __name__ == "__main__":
    configs = load_config()
    main(configs)