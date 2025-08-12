import time
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from optim import build_criterion, build_lr_scheduler, build_optimizer
from utils import AverageMeter, Checkpoint, ProgressMeter, Throughout
from .evaluator import Evaluator


class Trainer:
    def __init__(self, cfg, dataloaders, model, tensorboard_logger):
        super().__init__()
        self.cfg = cfg
        self.task_id = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.epochs = cfg.train.EPOCHS
        self.num_tasks = cfg.model.NUM_TASKS
        self.num_labels_per_task = cfg.model.NUM_LABELS_PER_TASK
        self.dataloaders = dataloaders
        self.tensorboard_logger = tensorboard_logger

        self.device = torch.device("cpu") if cfg.train.GPU is None else (
            torch.device(f"cuda:{cfg.train.GPU[0]}") if isinstance(cfg.train.GPU, list) else torch.device(f"cuda:{cfg.train.GPU}")
        )

        self.model = model.to(self.device)
        self.criterion = build_criterion(cfg.train.criterion)
        self.evaluator = Evaluator(cfg.test)
        self.scaler = GradScaler()
        self.checkpoint = Checkpoint(cfg.OUTPUT_DIR)

        # 将optimizer和lr_scheduler初始化为None，它们将在prepare_incremental_task中被创建
        self.optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_interval = None

        if cfg.train.CHECKPOINT:
            self.load_checkpoint(cfg.train.CHECKPOINT, cfg.train.CHECKPOINT_TASK_ID)

        if cfg.train.THROUGHOUT:
            self.throughout = Throughout()

    def train(self):
        if self.cfg.train.ONLY_VAL:
            logger.info("Debug model! Only run a validation function!")
            self.prepare_incremental_task(0)
            metrics = self.validate()
            self.display_metrics(metrics)
            return

        current_task_id = self.task_id
        result_matrix = np.zeros((self.num_tasks, self.num_tasks))

        # ======================= 探针变量初始化 =======================
        # 我们将存储Task 0某个关键参数的总和，作为它的"指纹"
        task0_param_fingerprint = None
        # ==========================================================

        for task_id in range(current_task_id, self.num_tasks):
            best_score = 0.0
            self.prepare_incremental_task(task_id)

            # ======================= 探针 #2：新任务开始前 =======================
            if task_id > 0 and task0_param_fingerprint is not None:
                try:
                    # 再次获取Task 0的同一个参数
                    param_to_check_before = self.model.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.lora_A_pools[0]
                    sum_before = param_to_check_before.sum().item()
                    logger.info(f"[探针 #2] Task {task_id} 开始前, Task 0参数指纹: {sum_before:.6f}")
                    if abs(sum_before - task0_param_fingerprint) > 1e-6:
                        logger.critical(f"!!! [探针 #2] 致命错误：Task 0的参数在任务过渡期间被改变了！期望值: {task0_param_fingerprint:.6f}, 实际值: {sum_before:.6f}")
                except Exception as e:
                    logger.error(f"[探针 #2] 无法获取参数: {e}")
            # ====================================================================

            for epoch in range(self.epochs):
                self.current_epoch = epoch
                self.train_loop()
                
                # 为了快速调试，我们可以在每个epoch后都验证
                metrics = self.validate()
                score = list(metrics[-1].values())[0].item()
                self.display_metrics(metrics)
                
                if best_score < score:
                    best_score = score
                    logger.info("New best score: {:.2f}.".format(best_score))
                    self.save_checkpoint(metrics, task_id=task_id, is_best=True)

            # 每个任务训练结束后，进行最终的验证和保存
            metrics = self.validate()
            self.display_metrics(metrics)
            self.save_checkpoint(metrics, task_id=task_id, is_best=False)

            for tid, metric in enumerate(metrics):
                result_matrix[task_id][tid] = metric[self.cfg.test.NAME]

            # ======================= 探针 #1 & #3：任务训练结束后 =======================
            if task_id == 0:
                # 探针 #1: 在Task 0训练结束后，记录下它的参数"指纹"
                try:
                    param_to_check_after_task0 = self.model.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.lora_A_pools[0]
                    task0_param_fingerprint = param_to_check_after_task0.sum().item()
                    logger.info(f"[探针 #1] Task 0 训练结束, 记录下参数指纹: {task0_param_fingerprint:.6f}")
                except Exception as e:
                    logger.error(f"[探针 #1] 无法获取参数: {e}")
            else:
                # 探针 #3: 在Task 1 (或更高) 训练结束后，检查Task 0的参数是否被"污染"
                try:
                    param_to_check_after_task1 = self.model.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.lora_A_pools[0]
                    sum_after = param_to_check_after_task1.sum().item()
                    logger.info(f"[探针 #3] Task {task_id} 训练结束, Task 0参数指纹: {sum_after:.6f}")
                    if abs(sum_after - task0_param_fingerprint) > 1e-6:
                        logger.critical(f"!!! [探针 #3] 致命错误：Task 0的参数在训练Task {task_id}时被改变了！期望值: {task0_param_fingerprint:.6f}, 实际值: {sum_after:.6f}")
                except Exception as e:
                    logger.error(f"[探针 #3] 无法获取参数: {e}")
            # ========================================================================

        ap, fg, last = self.evaluator.compute_cl_metric(result_matrix)
        logger.info(f"AP: {ap:.2f}, Forget: {fg:.2f}, Last: {last:.2f}")
        return ap

    def train_loop(self):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        total_losses = AverageMeter('TotalLoss', ':6.3f')
        cls_losses = AverageMeter('ClsLoss', ':6.3f')
        align_losses = AverageMeter('AlignLoss', ':6.3f')
        cons_losses = AverageMeter('ConsistLoss', ':6.3f')

        progress = ProgressMeter(
            len(self.train_dataloader),
            [batch_time, data_time, total_losses, cls_losses, align_losses, cons_losses],
            prefix="Epoch: [{}/{}]".format(self.current_epoch + 1, self.epochs),
        )

        self.model.train()
        task0_param_to_watch = None
        task0_param_before_sum = None

        if self.task_id > 0:
            try:
                task0_param_to_watch = self.model.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.lora_A_pools[0]
                task0_param_before_sum = task0_param_to_watch.sum().item()
            except Exception as e:
                logger.error(f"[探针] 无法获取要监控的Task 0参数: {e}")

        end = time.perf_counter()

        for it, batch in enumerate(self.train_dataloader):
            if self.cfg.train.THROUGHOUT:
                self.throughout.tick(len(batch["labels"]))

            self.current_iter += 1
            data_time.update(time.perf_counter() - end)

            self.optimizer.zero_grad()
            inputs = batch['inputs'].to(self.device)
            labels = batch['labels'].to(self.device)
            missing_types = batch['missing_types'].to(self.device)

            batch['inputs'] = inputs
            batch['labels'] = labels
            batch['missing_types'] = missing_types

            with autocast():
                model_outputs = self.model(batch)
                loss_dict = self.criterion(model_outputs)
                total_loss = loss_dict["total_loss"]

            total_losses.update(total_loss.item(), len(batch["labels"]))
            cls_losses.update(loss_dict["classification_loss"].item(), len(batch["labels"]))
            align_losses.update(loss_dict["alignment_loss"].item(), len(batch["labels"]))
            cons_losses.update(loss_dict["consistency_loss"].item(), len(batch["labels"]))

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 🔧 修复: 增强探针监控，每10个batch检查一次
            if task0_param_to_watch is not None and (it % 10 == 0):
                task0_param_after_sum = task0_param_to_watch.sum().item()
                if abs(task0_param_after_sum - task0_param_before_sum) > 1e-6:
                    logger.critical(f"!!! [探针] 致命错误：Task 0的参数在训练Task {self.task_id}时被修改了！")
                    logger.critical(f"训练前: {task0_param_before_sum:.8f}, 训练后: {task0_param_after_sum:.8f}")
                    logger.critical(f"差异: {abs(task0_param_after_sum - task0_param_before_sum):.8f}")
                    
                    # 🔧 额外检查：验证当前激活的任务ID是否正确
                    try:
                        current_vision_task_id = self.model.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.current_task_id
                        logger.critical(f"当前视觉适配器任务ID: {current_vision_task_id}, 期望: {self.task_id}")
                    except Exception as e:
                        logger.error(f"无法获取当前任务ID: {e}")

            if self.lr_scheduler and self.lr_scheduler_interval == "step":
                self.lr_scheduler.step()

            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            if (it + 1) % self.cfg.train.PRINT_FREQ == 0:
                progress.display(it + 1)
                self.tensorboard_logger.add_scalar(f"loss/task{self.task_id}_total", total_losses.avg, self.current_iter)
                self.tensorboard_logger.add_scalar(f"loss/task{self.task_id}_cls", cls_losses.avg, self.current_iter)
                self.tensorboard_logger.add_scalar(f"loss/task{self.task_id}_align", align_losses.avg, self.current_iter)
                self.tensorboard_logger.add_scalar(f"loss/task{self.task_id}_consist", cons_losses.avg, self.current_iter)

        if self.lr_scheduler and self.lr_scheduler_interval == "epoch":
            self.lr_scheduler.step()

    def prepare_incremental_task(self, task_id):
        """🔧 优化12: 简化任务准备日志"""
        logger.info(f"=== 准备任务 {task_id} ===")
        
        if hasattr(self.model, "set_active_task"):
            self.model.set_active_task(task_id)
        
        self._verify_task_switching(task_id)
        
        self.task_id = task_id
        self.valid_class_range = (0, (task_id + 1) * self.num_labels_per_task)
        self.prepare_dataloaders()
        self.reset_optimizer_and_scheduler()
        
        logger.info(f"=== 任务 {task_id} 准备完成 ===")

    def _verify_task_switching(self, expected_task_id):
        """🔧 修复: 验证任务切换是否成功"""
        try:
            # 检查第一层的query适配器
            first_layer_query = self.model.backbone.vilt.encoder.layer[0].attention.attention.query
            vision_adapter = first_layer_query.vision_adapter
            text_adapter = first_layer_query.text_adapter
            
            assert vision_adapter.current_task_id == expected_task_id, \
                f"视觉适配器任务ID错误: 期望{expected_task_id}, 实际{vision_adapter.current_task_id}"
            assert text_adapter.current_task_id == expected_task_id, \
                f"文本适配器任务ID错误: 期望{expected_task_id}, 实际{text_adapter.current_task_id}"
            
            # 验证参数冻结状态
            for task_idx in range(self.num_tasks):
                is_current_task = (task_idx == expected_task_id)
                
                # 检查LoRA参数的requires_grad状态
                lora_A_grad = vision_adapter.lora_A_pools[task_idx].requires_grad
                lora_B_grad = vision_adapter.lora_B_pools[task_idx].requires_grad
                
                assert lora_A_grad == is_current_task, \
                    f"任务{task_idx}的LoRA_A参数状态错误: 期望{is_current_task}, 实际{lora_A_grad}"
                assert lora_B_grad == is_current_task, \
                    f"任务{task_idx}的LoRA_B参数状态错误: 期望{is_current_task}, 实际{lora_B_grad}"
            
            logger.info(f"✅ 任务切换验证通过: 当前活跃任务 {expected_task_id}")
            
        except Exception as e:
            logger.error(f"❌ 任务切换验证失败: {e}")
            raise e

    def _verify_task_switching(self, expected_task_id):
        """🔧 优化9: 简化验证过程，只检查关键组件"""
        try:
            # 只检查第0层的query适配器作为代表
            first_layer_query = self.model.backbone.vilt.encoder.layer[0].attention.attention.query
            vision_adapter = first_layer_query.vision_adapter
            text_adapter = first_layer_query.text_adapter
            
            # 验证任务ID
            assert vision_adapter.current_task_id == expected_task_id, \
                f"任务ID错误: 期望{expected_task_id}, 实际{vision_adapter.current_task_id}"
            
            # 验证关键参数状态 - 只检查当前任务和前一个任务
            tasks_to_check = [expected_task_id]
            if expected_task_id > 0:
                tasks_to_check.append(expected_task_id - 1)
            
            for task_idx in tasks_to_check:
                is_current_task = (task_idx == expected_task_id)
                lora_A_grad = vision_adapter.lora_A_pools[task_idx].requires_grad
                
                assert lora_A_grad == is_current_task, \
                    f"任务{task_idx}参数状态错误: 期望{is_current_task}, 实际{lora_A_grad}"
            
            logger.info(f"✅ 任务切换验证通过: 当前活跃任务 {expected_task_id}")
            
        except Exception as e:
            logger.error(f"❌ 任务切换验证失败: {e}")
            raise e

    def reset_optimizer_and_scheduler(self):
        """🔧 修复2: 修复变量名错误"""
        # 获取所有可训练参数
        trainable_params = []
        layer_counts = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                
                # 统计每层参数数量
                if "layer." in name:
                    layer_num = name.split("layer.")[1].split(".")[0]
                    layer_counts[layer_num] = layer_counts.get(layer_num, 0) + param.numel()
        
        if not trainable_params:
            logger.error(f"❌ 任务 {self.task_id} 没有找到任何可训练参数!")
            raise ValueError(f"No trainable parameters found for task {self.task_id}")
        
        total_params = sum(p.numel() for p in trainable_params)
        active_layers = len(layer_counts)
        
        logger.info(f"✅ 任务 {self.task_id}: {len(trainable_params)} 个可训练参数 "
                   f"(总计 {total_params}, 涉及 {active_layers} 层)")
        
        # 🔧 修复3: 正确访问字典的键值对
        sample_layers = sorted(layer_counts.keys())[:3]
        if sample_layers:
            sample_info = ", ".join([f"Layer{k}:{layer_counts[k]}" for k in sample_layers])
            logger.info(f"参数分布样例: {sample_info}")
        
        # 创建优化器和调度器
        self.optimizer = build_optimizer(self.cfg.train.optimizer, trainable_params)
        
        num_training_steps = len(self.train_dataloader) * self.cfg.train.EPOCHS
        self.lr_scheduler = None if "lr_scheduler" not in self.cfg.train else build_lr_scheduler(
            self.cfg.train.lr_scheduler, self.optimizer, num_training_steps=num_training_steps
        )
        self.lr_scheduler_interval = None if "lr_scheduler" not in self.cfg.train else self.cfg.train.lr_scheduler.LR_SCHEDULER_INTERVAL



    def prepare_dataloaders(self):
        self.train_dataloader = self.dataloaders["train"][self.task_id]
        self.val_dataloaders = self.dataloaders["val"][self.task_id]

    @torch.no_grad()
    def _validate_one_dataloader(self, val_dataloader, task_id=0):
        """🔧 核心修复：验证时临时切换到目标任务"""
        logits, labels = [], []
        self.model.eval()

        # 🔧 关键修复1：记录当前激活的任务
        current_active_task = self.model.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.current_task_id
        
        # 🔧 关键修复2：如果要验证的任务不是当前激活的任务，临时切换
        need_task_switch = (task_id != current_active_task)
        if need_task_switch:
            logger.info(f"🔄 验证时临时切换: 任务{current_active_task} -> 任务{task_id}")
            self.model.backbone.set_active_task(task_id)

        try:
            # 验证时的探针检查
            if task_id == 0:
                try:
                    param_to_check_in_val = self.model.backbone.vilt.encoder.layer[0].attention.attention.query.vision_adapter.lora_A_pools[0]
                    sum_in_val = param_to_check_in_val.sum().item()
                    logger.info(f"[探针 #4] 验证Task {task_id}时, Task 0参数指纹: {sum_in_val:.6f}")
                except Exception as e:
                     logger.error(f"[探针 #4] 无法获取参数: {e}")

            for _, batch in tqdm(enumerate(val_dataloader), desc=f"task{task_id}", total=len(val_dataloader)):
                inputs_on_device = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                batch['inputs'] = inputs_on_device
                batch['labels'] = batch['labels'].to(self.device)
                batch['missing_types'] = batch['missing_types'].to(self.device)

                with autocast():
                    # 🔧 关键修复3：强制使用指定任务的专家系统
                    outputs = self.model(batch, force_task_id=task_id)

                logits.append(outputs["logits"].cpu())
                labels.append(batch["labels"].cpu())

        finally:
            # 🔧 关键修复4：验证完成后恢复原始任务状态
            if need_task_switch:
                logger.info(f"🔄 验证后恢复: 任务{task_id} -> 任务{current_active_task}")
                self.model.backbone.set_active_task(current_active_task)

        logits = torch.vstack(logits)[:, :self.valid_class_range[1]]
        if self.cfg.data.MULTI_LABEL:
            labels = torch.vstack(labels)[:, :self.valid_class_range[1]]
        else:
            labels = torch.hstack(labels)

        return self.evaluator({
            "logits": logits,
            "labels": labels,
        }, num_classes=labels.shape[1] if self.cfg.data.MULTI_LABEL else None)

    @torch.no_grad()
    def _validate_tasks(self, dataloaders, num_tasks):
        return [self._validate_one_dataloader(dataloaders[task_id], task_id) for task_id in range(num_tasks)]

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        return self._validate_tasks(self.val_dataloaders, self.task_id + 1)

    @torch.no_grad()
    def test(self, dataloaders, task_id):
        self.prepare_incremental_task(task_id)
        self.model.eval()
        return self._validate_tasks(dataloaders, task_id + 1)

    def save_checkpoint(self, metrics, task_id=None, is_best=False):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict(),
            "current_epoch": self.current_epoch,
            "metrics": metrics,
            "task_id": task_id,
        }
        if self.lr_scheduler:
            ckpt.update({"lr_scheduler": self.lr_scheduler.state_dict()})
        self.checkpoint.save(**ckpt, is_best=is_best)

    def load_checkpoint(self, checkpoint_path, task_id=None):
        """🔧 修复3: 增强的checkpoint加载逻辑"""
        logger.info(f"正在加载checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            logger.error(f"无法读取checkpoint文件: {e}")
            raise e
        
        # 确定任务ID
        if task_id is None:
            if "task_id" in checkpoint:
                task_id = checkpoint["task_id"]
            else:
                logger.warning("checkpoint中没有task_id信息，假设为task 0")
                task_id = 0
        
        logger.info(f"准备加载任务 {task_id} 的checkpoint")
        
        # 🔧 修复4: 在加载模型参数前，确保模型结构正确
        if hasattr(self.model, "set_active_task"):
            self.model.set_active_task(task_id)
            logger.info(f"已将模型切换到任务 {task_id}")
        
        # 加载模型参数
        try:
            if "model" in checkpoint:
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model"], strict=False)
                if missing_keys:
                    logger.warning(f"加载时缺失 {len(missing_keys)} 个键")
                if unexpected_keys:
                    logger.warning(f"加载时忽略 {len(unexpected_keys)} 个多余键")
                logger.info("✅ 模型参数加载成功")
            else:
                logger.error("checkpoint中没有模型参数")
                raise KeyError("No 'model' key found in checkpoint")
        except Exception as e:
            logger.error(f"模型参数加载失败: {e}")
            raise e
        
        logger.info(f"✅ 成功加载checkpoint (任务ID: {task_id})")

    def display_metrics(self, metrics):
        s = "Metrics:\n" + "=" * 50 + "\n"
        for task_id, metric in enumerate(metrics):
            for k, v in metric.items():
                s += f"Task {task_id} {k}: {v:.2f}\n"
        s += "=" * 50
        logger.info(s)


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch