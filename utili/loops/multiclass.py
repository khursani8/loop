from tqdm import tqdm
import numpy as np
import torch
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp16",gradient_accumulation_steps=1)

clip_percentile = 10
def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def train_once(cfg,data,model,optimizer,loss_func,grad_history,scheduler=None,ema=None):
    losses = []
    model.train()
    model, data,optimizer = accelerator.prepare(model, data,optimizer)
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
        obs_grad_norm = _get_grad_norm(model)
        grad_history.append(obs_grad_norm)
        if len(grad_history) > 150:
            clip_value = np.percentile(grad_history, clip_percentile)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)

        if cfg.sam_optimizer:
            optimizer.optimizer.first_step(zero_grad=True)
            outputs = model(images)
            loss_func(outputs, targets).backward()
            optimizer.optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
        scheduler.step()
        ema.update()
        out = {"loss":l}
        if scheduler:
            lr = scheduler.get_last_lr()[0]
            out["lr"] = round(lr,6)
        tbar.set_postfix(out)
    return {
        "losses":losses,
        "grad_history":grad_history
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

    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
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