from tqdm import tqdm
import numpy as np
import torch
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp16")

def train_once(cfg,data,model,optimizer,loss_func,scheduler=None,ema=None):
    losses = []
    model.train()
    model, optimizer, data = accelerator.prepare(model, optimizer, data)
    tbar = tqdm(enumerate(data),total=len(data))
    for batch_idx, batch in tbar:
        cfg.iter += 1
        if cfg.unfreeze_iter == cfg.iter:
            for params in model.parameters():
                params.requires_grad = True
            print("unfreeze model")
        images = batch["image"]
        targets = batch["target"]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = loss_func(outputs, targets)
        l = loss.item()
        losses.append(l)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        ema.update()
        out = {"loss":l}
        if scheduler:
            lr = scheduler.get_last_lr()[0]
            out["lr"] = round(lr,6)
        tbar.set_postfix(out)
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
    model.eval()
    preds = []
    model, data = accelerator.prepare(model, data)
    for batch in tqdm(data):
        images = batch["image"]
        outputs = model(images)
        preds.append(outputs)
    preds = torch.cat(preds)
    return preds.cpu()