from tqdm import tqdm
import numpy as np
import torch
from accelerate import Accelerator
accelerator = Accelerator()

def train_once(cfg,data,model,optimizer,loss_func,scheduler=None):
    losses = []
    model.train()
    model, optimizer, data = accelerator.prepare(model, optimizer, data)
    tbar = tqdm(enumerate(data),total=len(data))
    for batch_idx, batch in tbar:
        images = batch["image"]
        targets = batch["target"]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = loss_func(outputs, targets)
        l = loss.item()
        out = {"loss":l}
        tbar.set_postfix(out)
        losses.append(l)

        loss.backward()
        optimizer.step()
        scheduler.step()
    return {
        "losses":losses
    }


@torch.no_grad()
def validate_once(cfg,data,model,loss_func,metric_func):
    losses = []
    preds = []
    labels = []
    model.eval()

    model, data = accelerator.prepare(model, data)
    tbar = tqdm(enumerate(data),total=len(data))
    for idx,batch in tbar:
        images = batch["image"]
        targets = batch["target"]

        outputs = model(images)
        preds.append(outputs)
        labels.append(targets)
        loss = loss_func(outputs, targets)
        l = loss.item()
        out = {"loss":l}
        tbar.set_postfix(out)
        losses.append(l)

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    metric = metric_func(preds,labels)

    return {
        "losses": losses,
        "metric": metric
    }

@torch.no_grad()
def get_preds(data,model):
    preds = []
    model, data = accelerator.prepare(model, data)
    for batch in tqdm(data):
        images = batch["image"]
        outputs = model(images)
        preds.append(outputs)
    preds = torch.cat(preds)
    return preds.cpu()