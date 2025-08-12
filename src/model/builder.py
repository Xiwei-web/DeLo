# from .rebq import RebQ

# # 根据配置文件自动选择并创建模型对象，方便后续训练或测试流程调用
# def build_model(cfg):
#     model_name = cfg.NAME.lower()
#     if model_name == "rebq":
#         return RebQ(cfg)
#     else:
#         raise ValueError(f"Cannot find model name: {cfg.NAME}!")
# src/model/builder.py

# from .rebq import RebQ
from .rebq_mixlora import RebQMixLoRA # 新增：导入我们的新模型

# 根据配置文件自动选择并创建模型对象，方便后续训练或测试流程调用
def build_model(cfg):
    model_name = cfg.NAME.lower()
    if model_name == "rebq-mixlora": # 新增：为我们的新模型添加分支
        return RebQMixLoRA(cfg)