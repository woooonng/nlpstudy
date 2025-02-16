import wandb 
from tqdm import tqdm
import os

import torch
import torch.nn
import omegaconf
from omegaconf import OmegaConf

from utils import load_config, set_logger, setting_seed
from model import EncoderForClassification
from data import get_dataloader

# torch.cuda.set_per_process_memory_fraction(11/24) -> 김재희 로컬과 신입생 로컬의 vram 맞추기 용도. 과제 수행 시 삭제하셔도 됩니다. 
# model과 data에서 정의된 custom class 및 function을 import합니다.
import ast


def train_iter(model, inputs, optimizer, device, iters, use_wandb):
    inputs = {key : value.to(device) for key, value in inputs.items()}

    model.train()
    outputs = model(**inputs)
    loss = outputs['loss']

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = calculate_accuracy(outputs['logits'], inputs['label'])
    if use_wandb:
        wandb.log({
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': loss.item(),
            'train_accuracy': accuracy
        },
        step=iters)
    return loss.item(), accuracy

def valid_iter(model, inputs, device):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        loss = outputs['loss']
        accuracy = calculate_accuracy(outputs['logits'], inputs['label'])  
    return loss.item(), accuracy

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def main(configs : omegaconf.DictConfig) :
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Set seed
    setting_seed(configs.SEED)
    
    # Make the directories for results
    savedir = configs.RESULT.savedir
    cpdir = os.path.join(savedir, "checkpoint")
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(cpdir, exist_ok=True)

    # Set logger
    configs.LOG.file_path = configs.EXP_NAME
    _logger = set_logger(configs.LOG.dir_path, configs.LOG.file_path)

    # Log
    _logger.info(f"[DEVICE] {device}")
    _logger.info(OmegaConf.to_yaml(configs))
    
    # Use wandb
    if configs.TRAIN.use_wandb:
        wandb.init(name=configs.EXP_NAME, 
                   project=configs.TRAIN.wandb_project, 
                   config=OmegaConf.to_container(configs))
    
    # Load model
    model = EncoderForClassification(configs.MODEL).to(device)

    # Load data
    train_loader = get_dataloader(configs.DATASET, split='train')
    val_loader = get_dataloader(configs.DATASET, split='val')

    # Set optimizer
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr = configs.OPTIMIZER.lr,
        betas=tuple(map(float, ast.literal_eval(configs.OPTIMIZER.betas))),
        weight_decay=configs.OPTIMIZER.weight_decay
    )


    # Train & validation for each epoch
    best_acc = 0
    best_iters = 0
    best_epoch = 0
    best_total_iters = 0
    for epoch in range(configs.TRAIN.max_epoch):
        base_iters = len(train_loader) * epoch
        for iters, inputs in enumerate(train_loader, start=1):
            total_iters = base_iters + iters
            train_loss, train_accuracy = train_iter(model, inputs, optimizer, device, total_iters, configs.TRAIN.use_wandb)

            if iters == 1 or iters % configs.LOG.log_interval == 0:
                _logger.info(f"Epoch-{epoch+1}, Step-{iters}, loss-{train_loss:.2f}, acc-{train_accuracy*100:.2f}%, memory-{torch.cuda.memory_reserved(device)/(1000**3):.2f}GiB, lr-{optimizer.param_groups[0]['lr']:.6f}")

            # validation
            if iters == 1 or iters % configs.TRAIN.eval_interval == 0:
                val_loss = 0
                val_accuracy = 0
                for inputs in val_loader:
                    val_loss_iter, val_accuracy_iter = valid_iter(model, inputs, device)
                    val_loss += val_loss_iter
                    val_accuracy += val_accuracy_iter
                val_loss /= len(val_loader)
                val_accuracy /= len(val_loader)

                _logger.info(f"[EVAL] Epoch-{epoch+1}, Step-{iters}, loss-{val_loss:.2f}, acc-{val_accuracy*100:.2f}%")

                # use wandb
                if configs.TRAIN.use_wandb:
                    wandb.log({
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                    }, step=total_iters)

                # best model      
                if val_accuracy > best_acc:
                    # torch.save(model.state_dict(), os.path.join(savedir, f"{configs.MODEL.model_name}_best.pt"))
                    _logger.info(f"[BEST!] Epoch-{epoch+1}, Step-{iters}, total_iters-{total_iters}, Best accuracy: {best_acc*100:.2f}% to {val_accuracy*100:.2f}%")
                    best_acc = val_accuracy
                    best_iters = iters
                    best_epoch = epoch + 1
                    best_total_iters = total_iters

            torch.save(model.state_dict(), os.path.join(cpdir, f"{configs.MODEL.model_name}_latest_model.pt"))

    _logger.info(f"[END] Best model: Epoch-{best_epoch}, Step-{best_iters}, Total_iters-{best_total_iters}, Acc-{best_acc*100:.2f}%")
    
if __name__ == "__main__" :
    configs = load_config()
    main(configs)