from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
import omegaconf

class EncoderForClassification(nn.Module):
    def __init__(self, model_config : omegaconf.DictConfig):
        super().__init__()
        valid_model_name = ['bert', 'modern_bert']
        if model_config.model_name not in valid_model_name:
            raise ValueError(f"'{model_config.model_name} is not supported. Choose one from {valid_model_name}'") 
        
        if model_config.model_name == 'bert':
            self.model = AutoModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
        elif model_config.model_name == 'modern_bert':
            config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
            config.deterministic_flash_attn = False
            config._attn_implementation = 'eager'
            self.model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", config=config)

        self.classifier = nn.Linear(model_config.hidden_size, model_config.num_labels)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor, label : torch.Tensor, token_type_ids : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs : 
            input_ids : (batch_size, max_seq_len)
            attention_mask : (batch_size, max_seq_len)
            token_type_ids : (batch_size, max_seq_len) # only for BERT
            label : (batch_size)
        Outputs :
            logits : (batch_size, num_labels)
            loss : (1)
        """
        if token_type_ids is not None and token_type_ids.any():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_tokens = output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_tokens)
        loss = self.loss(logits, label)
        outputs = {
            'logits': logits,
            'loss': loss
        }
        return outputs
        
