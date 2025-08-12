# from loguru import logger


# def merge_cfg(cfg, params):
#     if "cfg.train.optimizer.LEARNING_RATE" in params.keys():
#         cfg.train.optimizer.LEARNING_RATE = params["cfg.train.optimizer.LEARNING_RATE"]
#         logger.info(f"Reset cfg.train.optimizer.LEARNING_RATE to {params['cfg.train.optimizer.LEARNING_RATE']}")
#     if "cfg.train.BATCH_SIZE" in params.keys():
#         cfg.train.BATCH_SIZE = params["cfg.train.BATCH_SIZE"]
#         logger.info(f"Reset cfg.train.BATCH_SIZE to {params['cfg.train.BATCH_SIZE']}")
#     if "cfg.model.prompt.LENGTH" in params.keys():
#         cfg.model.prompt.LENGTH = params["cfg.model.prompt.LENGTH"]
#         logger.info(f"Reset cfg.model.prompt.LENGTH to {params['cfg.model.prompt.LENGTH']}")
#     if "cfg.model.prompt.POOL_SIZE" in params.keys():
#         cfg.model.prompt.POOL_SIZE = params["cfg.model.prompt.POOL_SIZE"]
#         logger.info(f"Reset cfg.model.prompt.POOL_SIZE to {params['cfg.model.prompt.POOL_SIZE']}")
#     if "cfg.model.prompt.LAYERS" in params.keys():
#         cfg.model.prompt.LAYERS = params["cfg.model.prompt.LAYERS"]
#         logger.info(f"Reset cfg.model.prompt.LAYERS to {params['cfg.model.prompt.LAYERS']}")
#     if "cfg.model.prompt.ALPHA" in params.keys():
#         cfg.model.prompt.ALPHA = params["cfg.model.prompt.ALPHA"]
#         logger.info(f"Reset cfg.model.prompt.ALPHA to {params['cfg.model.prompt.ALPHA']}")

#     return cfg


# src/utils/hparams.py

from loguru import logger

def merge_cfg(cfg, params):
    # --- General Hyperparameters ---
    if "cfg.train.optimizer.LEARNING_RATE" in params.keys():
        cfg.train.optimizer.LEARNING_RATE = params["cfg.train.optimizer.LEARNING_RATE"]
        logger.info(f"Reset cfg.train.optimizer.LEARNING_RATE to {params['cfg.train.optimizer.LEARNING_RATE']}")
        
    if "cfg.train.BATCH_SIZE" in params.keys():
        cfg.train.BATCH_SIZE = params["cfg.train.BATCH_SIZE"]
        logger.info(f"Reset cfg.train.BATCH_SIZE to {params['cfg.train.BATCH_SIZE']}")

    # --- RebQ-MixLoRA Specific Hyperparameters ---
    # These keys should match what we define in our new rebq_mixlora.yaml config file
    
    if "cfg.model.LORA_R" in params.keys():
        cfg.model.LORA_R = params["cfg.model.LORA_R"]
        logger.info(f"Reset cfg.model.LORA_R to {params['cfg.model.LORA_R']}")
        
    if "cfg.model.LORA_E" in params.keys():
        cfg.model.LORA_E = params["cfg.model.LORA_E"]
        logger.info(f"Reset cfg.model.LORA_E to {params['cfg.model.LORA_E']}")

    if "cfg.model.LORA_ALPHA" in params.keys():
        cfg.model.LORA_ALPHA = params["cfg.model.LORA_ALPHA"]
        logger.info(f"Reset cfg.model.LORA_ALPHA to {params['cfg.model.LORA_ALPHA']}")

    if "cfg.train.criterion.LAMBDA_ALIGN" in params.keys():
        cfg.train.criterion.LAMBDA_ALIGN = params["cfg.train.criterion.LAMBDA_ALIGN"]
        logger.info(f"Reset cfg.train.criterion.LAMBDA_ALIGN to {params['cfg.train.criterion.LAMBDA_ALIGN']}")

    if "cfg.train.criterion.LAMBDA_CONSISTENCY" in params.keys():
        cfg.train.criterion.LAMBDA_CONSISTENCY = params["cfg.train.criterion.LAMBDA_CONSISTENCY"]
        logger.info(f"Reset cfg.train.criterion.LAMBDA_CONSISTENCY to {params['cfg.train.criterion.LAMBDA_CONSISTENCY']}")

    return cfg