# # 文件: src/model/mixlora_layer.py (最终修复版 - 回归物理分区)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from loguru import logger

# class MixLoRALayer(nn.Module):
#     def __init__(self, in_features: int, out_features: int, r: int, E: int, num_tasks: int, alpha: float = 1.0):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.r = r
#         self.E = E
#         self.num_tasks = num_tasks
#         self.alpha = alpha
#         self.current_task_id = 0

#         # ======================= 核心修改：回归物理分区架构 =======================
#         # 为每个任务创建独立的、物理隔离的LoRA A/B因子池
#         # 每个池的大小都是 E
#         self.lora_A_pools = nn.ParameterList([
#             nn.Parameter(torch.randn(E, r, self.in_features)) for _ in range(num_tasks)
#         ])
#         self.lora_B_pools = nn.ParameterList([
#             nn.Parameter(torch.zeros(E, self.out_features, r)) for _ in range(num_tasks)
#         ])
        
#         # 为每个任务创建独立的路由器
#         self.ifs_router_A_list = nn.ModuleList([nn.Linear(self.in_features, E) for _ in range(num_tasks)])
#         self.ifs_router_B_list = nn.ModuleList([nn.Linear(self.in_features, E) for _ in range(num_tasks)])
#         self.cfs_router_B_weights_list = nn.ParameterList([
#             nn.Parameter(torch.randn(r, self.in_features, E)) for _ in range(num_tasks)
#         ])
#         # =======================================================================
        
#         # Query-Key 记忆模块保持不变
#         self.register_buffer('task_keys', torch.zeros(num_tasks, self.in_features))
#         self.ema_momentum = 0.99

#     # 恢复 set_active_task 方法
#     def set_active_task(self, task_id: int):
#         self.current_task_id = task_id
#         for i in range(self.num_tasks):
#             is_active = (i == task_id)
#             # 冻结/解冻所有与任务相关的参数
#             self.lora_A_pools[i].requires_grad = is_active
#             self.lora_B_pools[i].requires_grad = is_active
#             self.cfs_router_B_weights_list[i].requires_grad = is_active
#             for param in self.ifs_router_A_list[i].parameters():
#                 param.requires_grad = is_active
#             for param in self.ifs_router_B_list[i].parameters():
#                 param.requires_grad = is_active

#     @torch.no_grad()
#     def _update_task_key(self, query_signal: torch.Tensor):
#         # 此方法不变
#         batch_key = query_signal.mean(dim=0)
#         old_key = self.task_keys[self.current_task_id]
#         self.task_keys[self.current_task_id] = self.ema_momentum * old_key + (1 - self.ema_momentum) * batch_key

#     def forward(self, x: torch.Tensor, query_signal: torch.Tensor) -> torch.Tensor:
#         # forward的逻辑现在变得更简单，因为它依赖于 set_active_task
#         task_id_to_use = self.current_task_id
#         force_task_id = getattr(x, 'force_task_id', None)

#         if not self.training: # 评估模式
#             if force_task_id is not None:
#                 task_id_to_use = force_task_id # 强制使用指定的专家
#             else: # 自动匹配
#                 with torch.no_grad():
#                     task_keys_on_device = self.task_keys.to(query_signal.device)
#                     similarities = F.cosine_similarity(query_signal.unsqueeze(1), task_keys_on_device.unsqueeze(0), dim=-1)
#                     best_task_ids = torch.argmax(similarities, dim=1)
#                     task_id_to_use = torch.mode(best_task_ids).values.item()
#         elif self.training:
#              self._update_task_key(query_signal.detach())
        
#         # 根据决策出的 task_id_to_use，选择激活的组件
#         active_A_pool = self.lora_A_pools[task_id_to_use]
#         active_B_pool = self.lora_B_pools[task_id_to_use]
#         active_ifs_router_A = self.ifs_router_A_list[task_id_to_use]
#         active_ifs_router_B = self.ifs_router_B_list[task_id_to_use]
#         active_cfs_router_B_weights = self.cfs_router_B_weights_list[task_id_to_use]

#         # 后续的路由和计算逻辑与之前完全相同
#         g_A_scores = active_ifs_router_A(query_signal)
#         g_A_indices = torch.topk(g_A_scores, self.r, dim=-1).indices
#         lora_A = active_A_pool[g_A_indices]

#         g_B_scores_ifs = active_ifs_router_B(query_signal)
#         # 注意：这里的cfs权重维度已经适配了物理分区模式
#         g_B_scores_cfs = torch.einsum('bri,rie->be', lora_A.detach(), active_cfs_router_B_weights).sum(dim=1)
        
#         g_B_scores = g_B_scores_ifs + g_B_scores_cfs
#         g_B_indices = torch.topk(g_B_scores, self.r, dim=-1).indices
#         lora_B = active_B_pool[g_B_indices]
        
#         after_A = torch.einsum('bsi,bri->bsr', x, lora_A)
#         lora_delta = torch.einsum('bsr,bor->bso', after_A, lora_B)
        
#         return lora_delta * self.alpha


# 文件1: src/model/mixlora_layer.py (修复版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class MixLoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, E: int, num_tasks: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.E = E
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.current_task_id = 0

        # 为每个任务创建独立的、物理隔离的LoRA A/B因子池
        self.lora_A_pools = nn.ParameterList([
            nn.Parameter(torch.randn(E, r, self.in_features)) for _ in range(num_tasks)
        ])
        self.lora_B_pools = nn.ParameterList([
            nn.Parameter(torch.zeros(E, self.out_features, r)) for _ in range(num_tasks)
        ])
        
        # 为每个任务创建独立的路由器
        self.ifs_router_A_list = nn.ModuleList([nn.Linear(self.in_features, E) for _ in range(num_tasks)])
        self.ifs_router_B_list = nn.ModuleList([nn.Linear(self.in_features, E) for _ in range(num_tasks)])
        self.cfs_router_B_weights_list = nn.ParameterList([
            nn.Parameter(torch.randn(r, self.in_features, E)) for _ in range(num_tasks)
        ])
        
        # Query-Key 记忆模块
        self.register_buffer('task_keys', torch.zeros(num_tasks, self.in_features))
        self.ema_momentum = 0.99

        # 🔧 修复1: 添加调试标记
        self.debug_name = "unnamed_layer"
        
        # 🔧 修复2: 在初始化时就设置Task 0为激活状态
        self.set_active_task(0)

    def set_debug_name(self, name: str):
        """设置调试名称，方便追踪问题"""
        self.debug_name = name

    def set_active_task(self, task_id: int):
        """🔧 修复: 修复 sum() 函数的参数错误"""
        old_task_id = self.current_task_id
        self.current_task_id = task_id
        
        for i in range(self.num_tasks):
            is_active = (i == task_id)
            
            # 冻结/激活LoRA因子池
            self.lora_A_pools[i].requires_grad = is_active
            self.lora_B_pools[i].requires_grad = is_active
            self.cfs_router_B_weights_list[i].requires_grad = is_active
            
            # 冻结/激活路由器参数
            for param in self.ifs_router_A_list[i].parameters():
                param.requires_grad = is_active
            for param in self.ifs_router_B_list[i].parameters():
                param.requires_grad = is_active

        # 🔧 修复: 只有在第0层的query的vision适配器时才打印详细信息
        if self.debug_name == "layer_0_query_vision":
            # 🔧 修复: 将计算过程分步进行，避免 sum() 函数的参数错误
            
            # 计算当前任务的激活参数数量
            current_task_lora_params = (
                self.lora_A_pools[task_id].numel() + 
                self.lora_B_pools[task_id].numel() +
                self.cfs_router_B_weights_list[task_id].numel()
            )
            
            current_task_router_params = 0
            for param in self.ifs_router_A_list[task_id].parameters():
                current_task_router_params += param.numel()
            for param in self.ifs_router_B_list[task_id].parameters():
                current_task_router_params += param.numel()
                
            total_activated_params = current_task_lora_params + current_task_router_params
            
            # 计算其他任务的冻结参数数量
            total_frozen_params = 0
            for i in range(self.num_tasks):
                if i != task_id:
                    frozen_lora_params = (
                        self.lora_A_pools[i].numel() + 
                        self.lora_B_pools[i].numel() +
                        self.cfs_router_B_weights_list[i].numel()
                    )
                    
                    frozen_router_params = 0
                    for param in self.ifs_router_A_list[i].parameters():
                        frozen_router_params += param.numel()
                    for param in self.ifs_router_B_list[i].parameters():
                        frozen_router_params += param.numel()
                        
                    total_frozen_params += frozen_lora_params + frozen_router_params
            
            logger.info(f"[代表性适配器] 任务切换: {old_task_id} -> {task_id}, "
                       f"激活参数: {total_activated_params}, 冻结参数: {total_frozen_params}")


    @torch.no_grad()
    def _update_task_key(self, query_signal: torch.Tensor):
        """确保query_signal的维度处理正确"""
        try:
            batch_key = query_signal.mean(dim=0)
            old_key = self.task_keys[self.current_task_id]
            self.task_keys[self.current_task_id] = self.ema_momentum * old_key + (1 - self.ema_momentum) * batch_key
        except Exception as e:
            logger.error(f"Task key更新错误: {e}")
            logger.error(f"query_signal形状: {query_signal.shape}")

    def forward(self, x: torch.Tensor, query_signal: torch.Tensor) -> torch.Tensor:
        task_id_to_use = self.current_task_id
        force_task_id = getattr(x, 'force_task_id', None)

        if not self.training:  # 评估模式
            if force_task_id is not None:
                task_id_to_use = force_task_id
            else:  # 自动匹配最佳任务
                with torch.no_grad():
                    task_keys_on_device = self.task_keys.to(query_signal.device)
                    similarities = F.cosine_similarity(
                        query_signal.unsqueeze(1), 
                        task_keys_on_device.unsqueeze(0), 
                        dim=-1
                    )
                    best_task_ids = torch.argmax(similarities, dim=1)
                    task_id_to_use = torch.mode(best_task_ids).values.item()
        elif self.training:
            # 确保在训练时使用正确的任务ID
            assert task_id_to_use == self.current_task_id, \
                f"训练时任务ID不匹配: current={self.current_task_id}, using={task_id_to_use}"
            self._update_task_key(query_signal.detach())
        
        # 选择激活的组件
        active_A_pool = self.lora_A_pools[task_id_to_use]
        active_B_pool = self.lora_B_pools[task_id_to_use]
        active_ifs_router_A = self.ifs_router_A_list[task_id_to_use]
        active_ifs_router_B = self.ifs_router_B_list[task_id_to_use]
        active_cfs_router_B_weights = self.cfs_router_B_weights_list[task_id_to_use]

        # 路由和计算逻辑
        try:
            g_A_scores = active_ifs_router_A(query_signal)
            g_A_indices = torch.topk(g_A_scores, self.r, dim=-1).indices
            lora_A = active_A_pool[g_A_indices]

            g_B_scores_ifs = active_ifs_router_B(query_signal)
            g_B_scores_cfs = torch.einsum('bri,rie->be', lora_A.detach(), active_cfs_router_B_weights).sum(dim=1)
            
            g_B_scores = g_B_scores_ifs + g_B_scores_cfs
            g_B_indices = torch.topk(g_B_scores, self.r, dim=-1).indices
            lora_B = active_B_pool[g_B_indices]
            
            after_A = torch.einsum('bsi,bri->bsr', x, lora_A)
            lora_delta = torch.einsum('bsr,bor->bso', after_A, lora_B)
            
            return lora_delta * self.alpha
            
        except Exception as e:
            logger.error(f"MixLoRA forward计算错误: {e}")
            logger.error(f"输入形状: x={x.shape}, query_signal={query_signal.shape}")
            logger.error(f"任务ID: {task_id_to_use}, 当前任务: {self.current_task_id}")
            raise e