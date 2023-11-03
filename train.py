import json
import time
import random
import pickle
import gc, os, sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from thefuzz import fuzz
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from lavis.models.blip2_models.blip2 import disabled_train
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import Adafactor, get_scheduler
from models import CLIPGuidedVQA, BLIP2GuidedVQA
from sklearn.metrics import accuracy_score, f1_score


class GuidanceVQADataset(Dataset):
    def __init__(self, filename, kmode="cso", samples=-1, data=None):
        images, texts, guidance, labels, qid = [], [], [], [], []
        
        if data == None:
            data = [json.loads(line) for line in open(filename).readlines()]
            if samples != -1:
                data = data[:samples]
        
        if "aokvqa" in filename:
            objects = [line["objects_detailed"] for line in data]
        else:
            objects = [line["objects"] for line in data]
            
        captions = [" & ".join(line["captions"][:4]) for line in data]
        scene_graphs = [line["scene_graph"] for line in data]
        
        if "hint" in data[0]:
            hints = [line["hint"] for line in data]
        else:
            hints = [""] * len(data)
            
        if "lecture" in data[0]:
            lectures = [line["lecture"] for line in data]
        else:
            lectures = [""] * len(data)
            
        if "rationales" in data[0]:
            rationales = [" & ".join(line["rationales"][:3]) for line in data]
        else:
            rationales = [""] * len(data)
            
        if "explanations" in data[0]:
            explanations = [" & ".join(line["explanations"][:3]) for line in data]
        else:
            explanations = [""] * len(data)
        
        for o, c, s, h, l, r, e in zip(objects, captions, scene_graphs, hints, lectures, rationales, explanations):
            k = ""
            if "r" in kmode:
                if r != "":
                    k += "Rationale: {}; ".format(r)
            if "e" in kmode:
                if e != "":
                    k += "Explanation: {}; ".format(e)
            if "c" in kmode:
                if c != "":
                    k += "Caption: {}; ".format(c)
            if "s" in kmode:
                if s != "":
                    k += "Scene Graph: {}; ".format(s)
            if "o" in kmode:
                if o != "":
                    k += "Objects: {}; ".format(o)
            if "h" in kmode:
                if o != "":
                    k += "Hint: {}; ".format(h)
            if "l" in kmode:
                if o != "":
                    k += "Lecture: {}; ".format(l)
                
            guidance.append(" ".join(k[:-2].strip().split()))
        
        for k, instance in enumerate(data):
            instance_texts = []
            question = instance["text_input"]
            
            for j, a in enumerate(instance["choices"]):
                instance_texts.append("{} {}".format(question, a))
                
            if "vsr" in filename:
                instance["image_path"] = instance["image_path"].replace(
                    "/mnt/data_02tb/deep/.cache/lavis/coco/images", "/mnt/data1/deep/.cache/lavis/coco/images"
                )
                    
            images.append(Image.open(instance["image_path"]).convert("RGB"))
            texts.append(instance_texts)
            
            if instance["correct_choice_idx"]: 
                labels.append(instance["correct_choice_idx"])
            else: 
                labels.append(0)
            qid.append(instance["question_id"])
                
        assert len(images) == len(guidance), "Num Images: {}; Num Guidance: {}".format(len(images), len(guidance))
        self.images, self.texts, self.guidance, self.labels, self.qid = images, texts, guidance, labels, qid
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        s1, s2, s3, s4, s5 = self.images[index], self.texts[index], self.guidance[index], self.labels[index], self.qid[index]
        return s1, s2, s3, s4, s5
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def configure_dataloaders(args):
    "Prepare dataloaders"
    print (f"Using text guidance from: {args.kmode}")
    print ("Preparing Dataloaders.")
    
    train_dataset = GuidanceVQADataset(f"data/{args.dataset}/train_with_guidance.json", args.kmode, args.samples)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.bs, collate_fn=train_dataset.collate_fn)

    val_dataset = GuidanceVQADataset(f"data/{args.dataset}/val_with_guidance.json", args.kmode, args.samples)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.eval_bs, collate_fn=val_dataset.collate_fn)
    
    test_dataset = GuidanceVQADataset(f"data/{args.dataset}/test_with_guidance.json", args.kmode, args.samples)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_bs, collate_fn=test_dataset.collate_fn)

    return train_loader, val_loader, test_loader


def configure_optimizer(model, args):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    return optimizer


def configure_scheduler(optimizer, num_training_steps, args):
    "Prepare scheduler"
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )    
    return lr_scheduler


def metric_from_list(preds, labels):
    all_preds = [item for sublist in preds for item in sublist]
    all_labels = [item for sublist in labels for item in sublist]
    acc = round(100 * accuracy_score(all_labels, all_preds), 2)
    f1 = round(100 * f1_score(all_labels, all_preds, average="macro"), 2)
    return acc, f1, all_preds


def train_or_eval_model(model, dataloader, optimizer=None, split="Train"):
    losses, preds, labels, question_ids = [], [], [], []
    if split=="Train":
        model.train()
        if "blip2" in args.name:
            model.model.visual_encoder.eval()
    else:
        model.eval()
    
    for batch in tqdm(dataloader, leave=False):
        if split=="Train":
            optimizer.zero_grad()
            loss, pred, label, qid = model(batch)
        else:
            with torch.no_grad():
                loss, pred, label, qid = model(batch)
        
        preds.append(pred)
        labels.append(label)
        question_ids += qid
        
        if split=="Train":
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)
    acc, _, preds = metric_from_list(preds, labels)
    
    if split in ["Train", "Val"]:
        wandb.log({"{} Loss".format(split): avg_loss})
        wandb.log({"{} Accuracy".format(split): acc})
        
    return avg_loss, acc, preds
    
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate for transformers.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--warm-up-steps", type=int, default=0, help="Warm up steps.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--eval-bs", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs.")
    parser.add_argument("--dataset", default="", help="Which dataset to train and evaluate on.")
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples to train and evaluate on.")
    parser.add_argument("--kmode", default="cso", help="What guidance to use.")
    parser.add_argument("--name", default="openai/clip-vit-large-patch14", help="Which model.")
    parser.add_argument("--fusion", default="concat-image", help="How to fuse the additional guidance for BLIP2.")
    parser.add_argument("--combine", default="features", help="How to combine normal image-qa logits and guidance logits for BLIP2.")
    
    global args
    args = parser.parse_args()
    print(args)
    
    name = args.name
    kmode = args.kmode
    fusion = args.fusion
    combine = args.combine
    epochs = args.epochs
    dataset = args.dataset
    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    
    if dataset in ["science_qa", "iconqa"]:
        num_choices = 5
    elif dataset in ["vsr"]:
        num_choices = 2
    elif dataset in ["aokvqa"]:
        num_choices = 4
    
    if "clip" in name:
        model = CLIPGuidedVQA(name, num_choices).cuda()
        model.extend_position_embeddings()
    elif "blip2" in name:
        model = BLIP2GuidedVQA(name.split("/")[1], fusion, combine, num_choices).cuda()
        print ("Freezing visual encoder")
        for param in model.model.visual_encoder.parameters():
            param.requires_grad = False
        model.model.visual_encoder.train = disabled_train
        
    print ("Num trainable parameters in model: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    optimizer = configure_optimizer(model, args)
    
    train_loader, val_loader, test_loader = configure_dataloaders(args)
    
    exp_id = str(int(time.time()) + random.randint(0, 100000))
    vars(args)["exp_id"] = exp_id
    
    name = name.lower()
    path = f"saved/{dataset}/{exp_id}/"
    Path(f"saved/{dataset}/{exp_id}/").mkdir(parents=True, exist_ok=True)
    Path(f"saved/{dataset}/{exp_id}/val_preds/").mkdir(parents=True, exist_ok=True)
    Path(f"saved/{dataset}/{exp_id}/test_preds/").mkdir(parents=True, exist_ok=True)
    
    fname = f"saved/{dataset}/{exp_id}/args.txt"
    
    f = open(fname, "a")
    f.write(str(args) + "\n\n")
    f.close()
       
    Path(f"results/{dataset}/").mkdir(parents=True, exist_ok=True)
    lf_name = f"results/{dataset}/guidance-" + name.replace("/", "-") + ".txt"
    lf_buffer = str(args) + "\n\n"

    if "clip" in name:
        wandb.init(project=f"{dataset}-CLIP".capitalize())
        wandb.watch(model)
    elif "blip2" in name:
        wandb.init(project=f"{dataset}-BLIP2".capitalize())
        wandb.watch(model)
    
    for e in range(epochs):
        train_loss, train_acc, train_preds = train_or_eval_model(model, train_loader, optimizer, "Train")
        val_loss, val_acc, val_preds = train_or_eval_model(model, val_loader, split="Val")
        test_loss, test_acc, test_preds = train_or_eval_model(model, test_loader, split="Test")
                
        with open(path + "valid_preds_epoch_" + str(e+1) + ".txt", "w") as f:
            f.write("\n".join([str(vp) for vp in val_preds]))
            
        with open(path + "test_preds_epoch_" + str(e+1) + ".txt", "w") as f:
            f.write("\n".join([str(tp) for tp in test_preds]))
                
        x = "Epoch {}: Loss: Train {}; Val {}; Test {}".format(e+1, train_loss, val_loss, test_loss)
        y = "Accuracy: Train {}; Val {}; Test {}".format(train_acc, val_acc, test_acc)
            
        print (x)
        print (y)
        
        epoch_result = x + "\n" + y + "\n\n"
        lf_buffer += epoch_result
        
        f = open(fname, "a")
        f.write(epoch_result)
        f.close()
        
    lf = open(lf_name, "a")
    lf.write(lf_buffer + "-"*100 + "\n")
    lf.close()