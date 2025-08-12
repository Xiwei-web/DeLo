import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math

from transformers import ViltModel, ViltConfig
# 导入我们需要的官方基类
from transformers.models.vilt.modeling_vilt import (
    ViltLayer, 
    ViltAttention, 
    ViltSelfAttention
)

from .mixlora_layer import MixLoRALayer 
from loguru import logger
# 在MixLoRAWrapper中添加一个set_active_task方法
# class MixLoRAWrapper(nn.Module):
#     def __init__(self, linear_layer: nn.Linear, model_cfg):
#         super().__init__()
#         self.frozen_layer = linear_layer
#         num_tasks = model_cfg.NUM_TASKS
#         in_features, out_features = linear_layer.in_features, linear_layer.out_features
        
#         # 注意：这里我们引用的是重构后的MixLoRALayer
#         self.vision_adapter = MixLoRALayer(in_features, out_features, r=model_cfg.LORA_R, E=model_cfg.LORA_E, num_tasks=num_tasks, alpha=model_cfg.LORA_ALPHA)
#         self.text_adapter = MixLoRALayer(in_features, out_features, r=model_cfg.LORA_R, E=model_cfg.LORA_E, num_tasks=num_tasks, alpha=model_cfg.LORA_ALPHA)

#     def set_active_task(self, task_id: int):
#         self.vision_adapter.set_active_task(task_id)
#         self.text_adapter.set_active_task(task_id)

#     def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
#         # 步骤1：先用冻结层计算基础输出
#         frozen_output = self.frozen_layer(hidden_states)
#         base_output = frozen_output[0] if isinstance(frozen_output, tuple) else frozen_output

#         # 步骤2：获取查询信号
#         query_signals = getattr(hidden_states, 'query_signals', None)

#         # 关键修复：只有在query_signals完全不存在时才短路返回
#         # 在我们的框架中，无论是训练还是评估，query_signals都应该被创建和传递
#         if query_signals is None:
#             return frozen_output

#         # 步骤3：计算LoRA增量 (这部分逻辑不变)
#         q_v, q_t, is_text_missing = query_signals['vision'], query_signals['text'], query_signals['is_text_missing']
        
#         vision_lora_delta = self.vision_adapter(hidden_states, q_v)
        
#         # 确保q_v和q_t的形状一致，以用于torch.where
#         if q_v.dim() == 1 and q_t.dim() > 1:
#             q_v = q_v.unsqueeze(0).expand_as(q_t)

#         text_query_signal = torch.where(is_text_missing.unsqueeze(-1).expand_as(q_t), q_v, q_t)
#         text_lora_delta = self.text_adapter(hidden_states, text_query_signal)

#         # 步骤4：应用增量
#         final_output = base_output + vision_lora_delta + text_lora_delta

#         if isinstance(frozen_output, tuple):
#             return (final_output,) + frozen_output[1:]
#         return final_output
class MixLoRAWrapper(nn.Module):
    def __init__(self, linear_layer: nn.Linear, model_cfg, layer_name: str = "unknown"):
        super().__init__()
        self.frozen_layer = linear_layer
        self.layer_name = layer_name  # 🔧 修复5: 添加层标识
        num_tasks = model_cfg.NUM_TASKS
        in_features, out_features = linear_layer.in_features, linear_layer.out_features
        
        # 创建双模态适配器
        self.vision_adapter = MixLoRALayer(
            in_features, out_features, 
            r=model_cfg.LORA_R, E=model_cfg.LORA_E, 
            num_tasks=num_tasks, alpha=model_cfg.LORA_ALPHA
        )
        self.text_adapter = MixLoRALayer(
            in_features, out_features, 
            r=model_cfg.LORA_R, E=model_cfg.LORA_E, 
            num_tasks=num_tasks, alpha=model_cfg.LORA_ALPHA
        )
        
        # 🔧 修复6: 设置调试名称
        self.vision_adapter.set_debug_name(f"{layer_name}_vision")
        self.text_adapter.set_debug_name(f"{layer_name}_text")

    def set_active_task(self, task_id: int):
        """🔧 优化4: 只在第0层打印详细信息"""
        if "layer_0" in self.layer_name:
            logger.info(f"[{self.layer_name}] 切换到任务{task_id}")
        self.vision_adapter.set_active_task(task_id)
        self.text_adapter.set_active_task(task_id)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # 冻结层计算
        frozen_output = self.frozen_layer(hidden_states)
        base_output = frozen_output[0] if isinstance(frozen_output, tuple) else frozen_output

        # 获取查询信号
        query_signals = getattr(hidden_states, 'query_signals', None)
        if query_signals is None:
            return frozen_output

        # 计算LoRA增量
        q_v, q_t, is_text_missing = query_signals['vision'], query_signals['text'], query_signals['is_text_missing']
        
        vision_lora_delta = self.vision_adapter(hidden_states, q_v)
        
        # 处理文本缺失情况
        if q_v.dim() == 1 and q_t.dim() > 1:
            q_v = q_v.unsqueeze(0).expand_as(q_t)

        text_query_signal = torch.where(is_text_missing.unsqueeze(-1).expand_as(q_t), q_v, q_t)
        text_lora_delta = self.text_adapter(hidden_states, text_query_signal)

        # 应用增量
        final_output = base_output + vision_lora_delta + text_lora_delta

        if isinstance(frozen_output, tuple):
            return (final_output,) + frozen_output[1:]
        return final_output


class MixLoRAViltSelfAttention(ViltSelfAttention):
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None: attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None: attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return (context_layer, attention_probs) if output_attentions else (context_layer,)

class MixLoRAViltAttention(ViltAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attention = MixLoRAViltSelfAttention(config)
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]

class MixLoRAViltLayer(ViltLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = MixLoRAViltAttention(config)
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        attention_outputs = self.attention(self.layernorm_before(hidden_states), attention_mask, head_mask, output_attentions)
        hidden_states = attention_outputs[0] + hidden_states
        layer_output = self.output(self.intermediate(self.layernorm_after(hidden_states)), hidden_states)
        return (layer_output,) + attention_outputs[1:]


# class RebQMixLoRAVilt(nn.Module):
#     def __init__(self, model_cfg):
#         super().__init__()
#         self.model_cfg = model_cfg
        
#         # 1. 首先，只加载原始的配置对象，而不是整个模型
#         config = ViltConfig.from_pretrained("dandelin/vilt-b32-mlm")
        
#         # 2. 获取我们需要的最大长度
#         max_len_config = model_cfg.get("MAX_LENGTH", config.max_position_embeddings)

#         # 3. 检查并直接修改这个配置对象
#         if max_len_config > config.max_position_embeddings:
#             print(f"\n[INFO] Modifying config for max_position_embeddings: {config.max_position_embeddings} -> {max_len_config}\n")
#             config.max_position_embeddings = max_len_config
        
#         # 4. 使用我们修改过的、尺寸正确的config，去加载预训练模型
#         #    ignore_mismatched_sizes=True 允许库智能地处理尺寸不符的权重
#         self.vilt = ViltModel.from_pretrained(
#             "dandelin/vilt-b32-mlm",
#             config=config,
#             ignore_mismatched_sizes=True # 关键！
#         )
#         # ==============================================================================

#         # LoRA注入逻辑 (保持不变)
#         for layer in self.vilt.encoder.layer:
#             layer.attention.attention.query = MixLoRAWrapper(layer.attention.attention.query, model_cfg)
#             layer.attention.attention.key = MixLoRAWrapper(layer.attention.attention.key, model_cfg)
#             layer.attention.attention.value = MixLoRAWrapper(layer.attention.attention.value, model_cfg)
    

#     def set_active_task(self, task_id: int):
#         for layer in self.vilt.encoder.layer:
#             layer.attention.attention.query.set_active_task(task_id)
#             layer.attention.attention.key.set_active_task(task_id)
#             layer.attention.attention.value.set_active_task(task_id)
class RebQMixLoRAVilt(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        
        # 加载配置和模型...
        config = ViltConfig.from_pretrained("dandelin/vilt-b32-mlm")
        max_len_config = model_cfg.get("MAX_LENGTH", config.max_position_embeddings)

        if max_len_config > config.max_position_embeddings:
            print(f"\n[INFO] Modifying config for max_position_embeddings: {config.max_position_embeddings} -> {max_len_config}\n")
            config.max_position_embeddings = max_len_config
        
        self.vilt = ViltModel.from_pretrained(
            "dandelin/vilt-b32-mlm",
            config=config,
            ignore_mismatched_sizes=True
        )

        # 🔧 优化5: 简化LoRA注入日志
        total_layers = len(self.vilt.encoder.layer)
        logger.info(f"开始为 {total_layers} 层注入 MixLoRA (query, key, value)...")
        
        injection_success = 0
        for layer_idx, layer in enumerate(self.vilt.encoder.layer):
            layer_name = f"layer_{layer_idx}"
            
            try:
                # Query注入
                layer.attention.attention.query = MixLoRAWrapper(
                    layer.attention.attention.query, model_cfg, f"{layer_name}_query"
                )
                
                # Key注入
                layer.attention.attention.key = MixLoRAWrapper(
                    layer.attention.attention.key, model_cfg, f"{layer_name}_key"
                )
                
                # Value注入
                layer.attention.attention.value = MixLoRAWrapper(
                    layer.attention.attention.value, model_cfg, f"{layer_name}_value"
                )
                
                injection_success += 1
                
                # 🔧 优化6: 只为每3层打印一次进度
                if (layer_idx + 1) % 3 == 0 or layer_idx == total_layers - 1:
                    logger.info(f"进度: {layer_idx + 1}/{total_layers} 层完成")
                    
            except Exception as e:
                logger.error(f"❌ Layer {layer_idx} 注入失败: {e}")
        
        logger.info(f"✅ MixLoRA注入完成: {injection_success}/{total_layers} 层成功")
        
        # 快速验证 - 只检查关键组件
        self._quick_verify()

    def _quick_verify(self):
        """🔧 优化7: 快速验证，只检查第0层和最后一层"""
        test_layers = [0, len(self.vilt.encoder.layer) - 1]
        success_count = 0
        
        for layer_idx in test_layers:
            layer = self.vilt.encoder.layer[layer_idx]
            if (isinstance(layer.attention.attention.query, MixLoRAWrapper) and
                isinstance(layer.attention.attention.key, MixLoRAWrapper) and
                isinstance(layer.attention.attention.value, MixLoRAWrapper)):
                success_count += 1
            else:
                logger.error(f"❌ Layer {layer_idx} 验证失败")
        
        if success_count == len(test_layers):
            logger.info(f"✅ LoRA注入验证通过 (检查了第0层和最后一层)")
        else:
            raise RuntimeError(f"LoRA注入验证失败!")


    def set_active_task(self, task_id: int):
        """🔧 修复1: 修复任务切换中的错误"""
        logger.info(f"全局切换到任务 {task_id}")
        
        switch_errors = 0
        total_layers = len(self.vilt.encoder.layer)
        
        for layer_idx, layer in enumerate(self.vilt.encoder.layer):
            try:
                layer.attention.attention.query.set_active_task(task_id)
                layer.attention.attention.key.set_active_task(task_id)
                layer.attention.attention.value.set_active_task(task_id)
            except Exception as e:
                switch_errors += 1
                if switch_errors <= 3:  # 只报告前3个错误
                    logger.error(f"Layer {layer_idx} 任务切换失败: {e}")
                    # 🔧 添加详细错误信息
                    import traceback
                    logger.error(f"详细错误: {traceback.format_exc()}")
        
        if switch_errors == 0:
            logger.info(f"✅ 所有 {total_layers} 层已切换到任务 {task_id}")
        else:
            logger.error(f"❌ {switch_errors} 层任务切换失败")

    # ==================== 核心修复：明确的函数签名来接收所有参数 ====================
    def forward(self, 
            input_ids, 
            pixel_values, 
            attention_mask=None, 
            token_type_ids=None,
            pixel_mask=None,
            inputs_embeds=None,
            image_embeds=None,
            force_task_id: int = None):

        # 步骤 1: 将所有将要传递给 embedding 层的参数打包到一个字典中
        embedding_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "inputs_embeds": inputs_embeds,
            "image_embeds": image_embeds,
        }
    
        # embedding_output, updated_attention_mask = self.vilt.embeddings(...)
        try:
            # 步骤 3: 使用字典解包 `**` 的方式进行调用，这是最稳健的关键字参数传递方式
            embedding_output, updated_attention_mask = self.vilt.embeddings(**embedding_args)
        
        except TypeError as e:
            # 步骤 4: 如果依然报错，我们将捕获它，并打印出更详细的求救信息
            logger.critical("!!! 调用 self.vilt.embeddings 时捕获到 TypeError !!!")
            logger.critical(f"错误信息: {e}")
            # 再次打印参数信息，以防万一
            for key, value in embedding_args.items():
                logger.critical(f"崩溃时参数详情 -> {key}: {type(value)}")
            raise e # 重新抛出异常，让程序停止
        
        hidden_states = embedding_output
        extended_attention_mask = self.vilt.get_extended_attention_mask(updated_attention_mask, input_ids.shape)
        text_len = input_ids.shape[1]
        is_text_missing = (input_ids[:, 1:] == 0).all(dim=1)

        if force_task_id is not None:
            hidden_states.force_task_id = force_task_id
        
        # final_query_signals 初始化
        final_query_signals = {}
        for layer_module in self.vilt.encoder.layer:
            q_v = hidden_states[:, text_len:, :].mean(dim=1)
            q_t = hidden_states[:, 1:text_len, :].mean(dim=1)
            q_t[is_text_missing] = 0.0
            
            hidden_states.query_signals = {'vision': q_v, 'text': q_t, 'is_text_missing': is_text_missing}
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
            
            if hasattr(hidden_states, 'query_signals'):
                del hidden_states.query_signals
            
            # =================== 核心修改点 ===================
            # 无论是否在训练，我们都记录最后一层的query_signals
            # 这对于损失计算和调试都有好处
            if layer_module == self.vilt.encoder.layer[-1]:
                final_query_signals = {'vision': q_v, 'text': q_t}
            # ===============================================

        sequence_output = self.vilt.layernorm(hidden_states)
        
        if hasattr(hidden_states, 'force_task_id'):
            del hidden_states.force_task_id

        final_output = {
            "last_hidden_state": sequence_output,
            "pooler_output": self.vilt.pooler(sequence_output),
        }
    
        # 无论训练还是评估，都将最后一层的信号返回
        final_output["query_signals"] = final_query_signals
            
        return final_output