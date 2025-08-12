

# 文件: src/optim/criterion.py (最终诊断修复版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger # 导入logger以便打印

def build_criterion(cfg):
    criterion_name = cfg.NAME.lower()
    if criterion_name == "rebq-mixlora":
        return RebQMixLoRALoss(cfg)
    elif criterion_name == "ce":
        return nn.CrossEntropyLoss()
    elif criterion_name == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown criterion name: {cfg.NAME}")

class RebQMixLoRALoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_name = cfg.DATASET_NAME.lower()
        self.lambda_align = cfg.get("LAMBDA_ALIGN", 0.1)
        self.lambda_consistency = cfg.get("LAMBDA_CONSISTENCY", 0.1)

        if self.dataset_name == "upmc_food101_cmml":
            self.classification_loss = nn.CrossEntropyLoss()
        elif self.dataset_name == "mm_imdb_cmml":
            self.classification_loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown dataset name for criterion: {self.dataset_name}")
            
        self.consistency_loss = nn.KLDivLoss(reduction='batchmean')
        self.alignment_loss = nn.CosineEmbeddingLoss()

    def forward(self, model_outputs):
        logits = model_outputs["logits"]
        labels = model_outputs["labels"]
        L_c = self.classification_loss(logits, labels)

        L_align = torch.tensor(0.0, device=logits.device)
        L_consistency = torch.tensor(0.0, device=logits.device)

        if 'query_signals' in model_outputs and 'complete_indices' in model_outputs:
            complete_indices = model_outputs["complete_indices"]
            
            if len(complete_indices) > 0:
                # --- 对齐损失 L_align ---
                query_signals = model_outputs["query_signals"]
                q_v = query_signals['vision']
                q_t = query_signals['text']
                
                q_v_complete = q_v[complete_indices]
                q_t_complete = q_t[complete_indices]

                # ==================== 最终探针与修复 ====================
                # 使用.view(-1, 768)来强制确保张量是2D的，即使只有一个样本
                # 768是ViLT的hidden_size
                hidden_dim = q_v_complete.shape[-1]
                q_v_complete = q_v_complete.view(-1, hidden_dim)
                q_t_complete = q_t_complete.view(-1, hidden_dim)

                # logger.info(f"Alignment Loss Input Shapes: q_v={q_v_complete.shape}, q_t={q_t_complete.shape}")
                # ========================================================
                
                target = torch.ones(q_v_complete.size(0)).to(logits.device)
                L_align = self.alignment_loss(q_v_complete, q_t_complete, target)

                # --- 一致性损失 L_consistency ---
                if 'consistency_logits' in model_outputs:
                    consistency_logits = model_outputs["consistency_logits"]
                    original_logits_for_complete = logits[complete_indices]

                    if original_logits_for_complete.shape[0] == consistency_logits.shape[0]:
                        log_p_original = F.log_softmax(original_logits_for_complete, dim=-1)
                        p_consistency = F.softmax(consistency_logits, dim=-1)
                        L_consistency = self.consistency_loss(log_p_original, p_consistency)
        
        total_loss = L_c + self.lambda_align * L_align + self.lambda_consistency * L_consistency
        
        return {
            "total_loss": total_loss,
            "classification_loss": L_c,
            "alignment_loss": L_align,
            "consistency_loss": L_consistency
        }