import json
import os
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPProcessor, ViltProcessor

from .datasets import MmImdbCmmlDataset, UpmcFood101CmmlDataset

Image.MAX_IMAGE_PIXELS = 1000000000


def build_dataloaders(cfg, **kwargs):
    # Build processor
    processor_name = cfg.PROCESSOR.lower()
    if processor_name == "vilt" or processor_name == "rebq-mixlora": # 更名以匹配新模型
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    elif processor_name == "clip":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Cannot find processor named: {processor_name}")

    # Build dataset and data loader
    dataset_name = cfg.NAME.lower()
    assert dataset_name in ("upmc_food101_cmml", "mm_imdb_cmml")

    # Prepare all data
    task_data = prepare_task_data(cfg.DATA_DIR, dataset_name, cfg.NUM_TASKS)

    # <<--- 修改部分：将 collate_fn 的 training 参数移除，因为它不再需要根据训练状态来改变行为
    collate_fn = partial(our_collate_fn, dataset_name=dataset_name, processor=processor)

    if dataset_name == "upmc_food101_cmml":
        DatasetClass = UpmcFood101CmmlDataset
    elif dataset_name == "mm_imdb_cmml":
        DatasetClass = MmImdbCmmlDataset

    dataloaders = {"train": [], "val": []} # length: num_tasks
    for task_id in range(cfg.NUM_TASKS):
        train_dataset = DatasetClass(
            data=task_data[task_id]["train"],
            missing_params=cfg.missing_params,
            split="train",
            task_id=task_id,
            limited_data_ratio=cfg.LIMITED_TRAIN_DATA_RATIO,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            shuffle=None if kwargs["distributed"] else True,
            sampler=DistributedSampler(train_dataset, shuffle=True) if kwargs["distributed"] else None,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        dataloaders["train"].append(train_dataloader)

        # Construct val loaders from 0 to current task id
        val_dataloaders = []
        for tid in range(task_id+1):
            val_dataset = DatasetClass(
                data=task_data[tid]["val"],
                missing_params=cfg.missing_params,
                split="val",
                task_id=tid,
                limited_data_ratio=cfg.LIMITED_VAL_DATA_RATIO,
            )
            val_dataloaders.append(DataLoader(
                val_dataset,
                batch_size=cfg.TEST_BATCH_SIZE,
                sampler=DistributedSampler(val_dataset, shuffle=False) if kwargs["distributed"] else None,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
                collate_fn=collate_fn,
            ))
        dataloaders["val"].append(val_dataloaders)
        
    test_loaders = []
    for tid in range(cfg.NUM_TASKS):
        test_dataset = DatasetClass(
            data=task_data[tid]["test"],
            missing_params=cfg.missing_params,
            split="test",
            task_id=tid,
        )
        test_loaders.append(DataLoader(
            test_dataset,
            batch_size=cfg.TEST_BATCH_SIZE,
            sampler=DistributedSampler(test_dataset, shuffle=False) if kwargs["distributed"] else None,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
        ))
    dataloaders["test"] = test_loaders
    
    return dataloaders


def prepare_task_data(data_dir, dataset_name, num_tasks):
    """ Prepare all task data.
    """
    if dataset_name == "upmc_food101_cmml":
        json_filename = "UPMC-Food101-CMML.json"
    elif dataset_name == "mm_imdb_cmml":
        json_filename = "MM-IMDB-CMML.json"
    else:
        raise ValueError(f"Cannot find dataset name: {dataset_name}")
    with open(os.path.join(data_dir, json_filename), "r") as f:
        json_data = json.load(f)
    
    task_data = []
    for _ in range(num_tasks):
        task_data.append({"train": [], "val": [], "test": []})

    for item in json_data:
        item_task_id = item["task_id"]
        if item_task_id < num_tasks:
            # For debug, we only use a subset and set num_tasks=1
            item_split = item["split"]
            task_data[item_task_id][item_split].append({
                "image": os.path.join(data_dir, "images", item["image"]),
                "text": ". ".join(item["text"]) if isinstance(item["text"], list) else item["text"],
                "label": item["label"],
            })

    return task_data


# <<--- 修改部分：简化 our_collate_fn 函数
def our_collate_fn(data, dataset_name, processor):
    """
    A simplified collate function that batches data and prepares it for the model.
    The logic for creating artificial missing samples for consistency loss
    will be handled in the trainer, not here.
    """
    batch = {
        "images": [],
        "texts": [],
        "labels": [],
        "missing_types": [],
    }

    for item in data:
        batch["images"].append(item["image"])
        batch["texts"].append(item["text"])
        
        # Handle label conversion based on dataset type
        if dataset_name == "upmc_food101_cmml":
            # Labels will be converted to tensor all at once later
            batch["labels"].append(item["label"])
        elif dataset_name == "mm_imdb_cmml":
            # For multi-label, convert each to a tensor immediately
            batch["labels"].append(torch.tensor(item["label"]))
            
        batch["missing_types"].append(item["missing_type"])

    # Convert list of labels to a single tensor
    if dataset_name == "upmc_food101_cmml":
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
    elif dataset_name == "mm_imdb_cmml":
        batch["labels"] = torch.stack(batch["labels"]).float()
        
    # Convert missing types to a tensor for easier processing in the model
    batch["missing_types"] = torch.tensor(batch["missing_types"], dtype=torch.long)

    # Use the processor to convert images and texts into model inputs
    inputs = processor(
        text=batch["texts"],
        images=batch["images"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    
    #  # ==================== DEBUG探针 #1 ====================
    # print("\n--- [探针#1] Dataloader输出 ---")
    # if 'pixel_values' in inputs:
    #     print(f"图像张量 'pixel_values' 的实际形状: {inputs['pixel_values'].shape}")
    # print(f"文本张量 'input_ids' 的实际形状: {inputs['input_ids'].shape}")
    # print("-----------------------------------\n")
    # # ======================================================

    batch["inputs"] = inputs

    return batch