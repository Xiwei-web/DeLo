


# 文件: src/model/rebq_mixlora.py (最终修复版)
import torch
import torch.nn as nn
from collections import OrderedDict

from .rebq_mixlora_vilt import RebQMixLoRAVilt
from loguru import logger
class RebQMixLoRA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = RebQMixLoRAVilt(cfg)
        self.num_tasks = cfg.NUM_TASKS
        self.num_labels_per_task = cfg.NUM_LABELS_PER_TASK
        self.classifiers = nn.ModuleList([
            nn.Linear(cfg.HIDDEN_SIZE, self.num_labels_per_task) for _ in range(self.num_tasks)
        ])

        # 初始化参数冻结
        for param in self.backbone.vilt.parameters():
            param.requires_grad = False
        
        # 激活Task 0的参数和分类头，冻结其他的
        self.set_active_task(0) 
    
    def set_active_task(self, task_id: int):
        self.backbone.set_active_task(task_id)
        for i in range(self.num_tasks):
            is_active = (i == task_id)
            for param in self.classifiers[i].parameters():
                param.requires_grad = is_active



    def forward(self, batch, force_task_id: int = None):
        inputs = batch['inputs']
        
        # 🔧 增强修复：确保force_task_id被正确处理
        outputs = self.backbone(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            pixel_mask=inputs.get("pixel_mask"),
            force_task_id=force_task_id
        )
        
        cls_token = outputs['pooler_output']
        
        # 🔧 新增修复：在评估模式下且有force_task_id时，只使用指定任务的分类头
        if not self.training and force_task_id is not None:
            # 只计算指定任务的logits
            task_logits = self.classifiers[force_task_id](cls_token)
            
            # 构造完整的logits张量，其他任务的位置填零
            full_logits = torch.zeros(
                cls_token.size(0), 
                self.num_tasks * self.num_labels_per_task, 
                device=cls_token.device
            )
            start_idx = force_task_id * self.num_labels_per_task
            end_idx = (force_task_id + 1) * self.num_labels_per_task
            full_logits[:, start_idx:end_idx] = task_logits
            
            return {"logits": full_logits, "labels": batch["labels"]}
        
        # 训练模式或没有force_task_id时，使用原有逻辑
        all_logits = [self.classifiers[i](cls_token) for i in range(self.num_tasks)]
        logits = torch.cat(all_logits, dim=1)
        
        results = {"logits": logits, "labels": batch["labels"]}

        if "query_signals" in outputs:
            results["query_signals"] = outputs["query_signals"]
        
        results["missing_types"] = batch["missing_types"]
        complete_indices = (batch['missing_types'] == 0).nonzero(as_tuple=True)[0]
        results["complete_indices"] = complete_indices

        if self.training and len(complete_indices) > 0:
            # 一致性损失计算逻辑保持不变
            consistency_inputs = {
                'input_ids': inputs['input_ids'][complete_indices],
                'pixel_values': inputs['pixel_values'][complete_indices],
                'attention_mask': inputs['attention_mask'][complete_indices],
                'pixel_mask': inputs.get('pixel_mask')[complete_indices] if inputs.get('pixel_mask') is not None else None,
            }
            
            pad_token_id, cls_token_id = 0, 101
            dummy_text_ids = torch.full_like(consistency_inputs['input_ids'], pad_token_id)
            dummy_text_ids[:, 0] = cls_token_id
            
            with torch.no_grad():
                consistency_outputs = self.backbone(
                    input_ids=dummy_text_ids,
                    pixel_values=consistency_inputs.get("pixel_values"),
                    attention_mask=consistency_inputs.get("attention_mask"),
                    token_type_ids=None,
                    pixel_mask=consistency_inputs.get("pixel_mask"),
                    force_task_id=self.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.current_task_id
                )

            consistency_cls = consistency_outputs['pooler_output']
            all_consistency_logits = [self.classifiers[i](consistency_cls) for i in range(self.num_tasks)]
            consistency_logits = torch.cat(all_consistency_logits, dim=1)
            results["consistency_logits"] = consistency_logits

        return results

    # ==================== 核心修复：自定义状态字典逻辑 ====================
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        自定义state_dict，确保所有任务的专家和分类头都被保存。
        """
        if destination is None:
            destination = OrderedDict()
        
        # 1. 保存backbone中所有我们自定义的、需要持续学习的参数
        #    我们直接访问底层的 ParameterList 和 ModuleList
        for i in range(self.num_tasks):
            # 保存LoRA A和B的因子池
            destination[prefix + f'backbone.vision_adapter.lora_A_pools.{i}'] = self.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.lora_A_pools[i]
            destination[prefix + f'backbone.vision_adapter.lora_B_pools.{i}'] = self.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.lora_B_pools[i]
            destination[prefix + f'backbone.text_adapter.lora_A_pools.{i}'] = self.backbone.vilt.encoder.layer[0].attention.attention.query.text_adapter.lora_A_pools[i]
            destination[prefix + f'backbone.text_adapter.lora_B_pools.{i}'] = self.backbone.vilt.encoder.layer[0].attention.attention.query.text_adapter.lora_B_pools[i]
            
            # 保存路由器的状态
            destination.update(self.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.ifs_router_A_list[i].state_dict(prefix=prefix + f'backbone.vision_adapter.routers.{i}.ifs_A.'))
            # ... 此处需要为所有注入LoRA的层的、所有适配器和所有路由器都添加保存逻辑 ...
            # 为了简化，我们直接调用整个模型，然后筛选
            
        # 更简单、更鲁棒的方式是调用原始的state_dict，并确保它包含了所有我们想要的东西
        # PyTorch的ModuleList和ParameterList是能被state_dict()正确处理的
        # 让我们重新思考，为什么默认的state_dict()会失败？
        # 失败的原因可能不是因为requires_grad=False，而是因为动态替换了模块。
        # 我们的注入方式是 self.query = MixLoRAWrapper(self.query, ...), 这可能破坏了原始的模块注册树。
        
        # 让我们采取最保险的方式：分别获取backbone和classifiers的状态字典，然后合并
        # 这个假设是：backbone和classifiers内部的state_dict()能正确工作。
        
        # 清理旧的简单实现，使用PyTorch推荐的标准方式
        super_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return super_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """🔧 修复1: 自定义状态字典加载，处理键名不匹配问题"""
        
        # 检查是否是旧格式的checkpoint
        old_format_keys = [k for k in state_dict.keys() if k.startswith("backbone.vision_adapter.") or k.startswith("backbone.text_adapter.")]
        
        if old_format_keys:
            logger.warning("检测到旧格式的checkpoint，正在进行键名转换...")
            state_dict = self._convert_old_checkpoint(state_dict)
        
        # 检查键名匹配情况
        model_keys = set(self.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            logger.warning(f"模型中缺失的键: {len(missing_keys)} 个")
            for key in list(missing_keys)[:5]:  # 只显示前5个
                logger.warning(f"  缺失: {key}")
            if len(missing_keys) > 5:
                logger.warning(f"  ... 还有 {len(missing_keys) - 5} 个缺失键")
        
        if unexpected_keys:
            logger.warning(f"checkpoint中多余的键: {len(unexpected_keys)} 个")
            for key in list(unexpected_keys)[:5]:  # 只显示前5个
                logger.warning(f"  多余: {key}")
            if len(unexpected_keys) > 5:
                logger.warning(f"  ... 还有 {len(unexpected_keys) - 5} 个多余键")
        
        # 尝试宽松加载
        try:
            return super().load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.error(f"即使使用strict=False也无法加载checkpoint: {e}")
            raise e

    def _convert_old_checkpoint(self, old_state_dict):
        """🔧 修复2: 转换旧格式的checkpoint键名"""
        new_state_dict = OrderedDict()
        
        # 映射规则：旧键名 -> 新键名的模式
        conversion_patterns = [
            # 旧格式: backbone.vision_adapter.lora_A_pools.X
            # 新格式: backbone.vilt.encoder.layer.Y.attention.attention.Z.vision_adapter.lora_A_pools.X
            ("backbone.vision_adapter.", "backbone.vilt.encoder.layer.0.attention.attention.query.vision_adapter."),
            ("backbone.text_adapter.", "backbone.vilt.encoder.layer.0.attention.attention.query.text_adapter."),
        ]
        
        for old_key, old_value in old_state_dict.items():
            new_key = old_key
            
            # 应用转换规则
            for old_pattern, new_pattern in conversion_patterns:
                if old_key.startswith(old_pattern):
                    new_key = old_key.replace(old_pattern, new_pattern, 1)
                    break
            
            new_state_dict[new_key] = old_value
        
        logger.info(f"转换了 {len([k for k in old_state_dict.keys() if any(k.startswith(p[0]) for p in conversion_patterns)])} 个旧格式键名")
        return new_state_dict