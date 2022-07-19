from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from utili.utils import stt
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="no",gradient_accumulation_steps=1)

def train_once(cfg,data,model,optimizer,loss_func,scheduler=None,ema=None):
    losses = []
    model.train()
    model,data,optimizer = accelerator.prepare(model,data,optimizer)
    tbar = tqdm(enumerate(data),total=len(data))
    for idx,batch in tbar:
        if cfg.unfreeze_iter == cfg.iter:
            for params in model.parameters():
                params.requires_grad = True
            print("unfreeze model")
        audios = batch["audio"]
        audios_len = batch["audio_len"]
        targets = batch["target"]
        targets_len = batch["target_len"]

        optimizer.zero_grad()
        logits,logits_len = model(audios,audios_len) # B,T,F
        log_probs = F.log_softmax(logits, dim=2)
        ctc_input = log_probs.transpose(0, 1)  # (T,B,C)
        loss = loss_func(ctc_input, targets, logits_len, targets_len)
        l = loss.item()
        losses.append(l)
        accelerator.backward(loss)

        if cfg.sam_optimizer:
            optimizer.optimizer.first_step(zero_grad=True)
            logits,logits_len = model(audios,audios_len)
            log_probs = F.log_softmax(logits, dim=2)
            ctc_input = log_probs.transpose(0, 1)  # (T,B,C)
            loss = loss_func(ctc_input, targets, logits_len, targets_len)
            loss.mean().backward()
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
        if idx == 0:
            output  = stt.decode(log_probs[0].argmax(-1).type(torch.int),logits_len[0],cfg.labels)
            label   = stt.decode(targets[0],targets_len[0],cfg.labels)
            print(f"{label} | {output}")
    return {
        "losses":losses
    }

@torch.no_grad()
def validate_once(cfg,data,model,loss_func,met_func):
    losses = []
    model.eval()
    model,data = accelerator.prepare(model,data)
    tbar = tqdm(data,total=len(data))
    preds = []
    labels = []

    for batch in tbar:
        audios = batch["audio"]
        audios_len = batch["audio_len"]
        targets = batch["target"]
        targets_len = batch["target_len"]

        logits,logits_len = model(audios,audios_len) # B,T,F
        log_probs = F.log_softmax(logits, dim=2)
        ctc_input = log_probs.transpose(0, 1)  # (T,B,C)
        loss = loss_func(ctc_input, targets, logits_len, targets_len)
        l = loss.item()
        losses.append(l)
        outputs  = stt.decodes(log_probs.argmax(2).type(torch.int),logits_len,cfg.labels)
        lbls  = stt.decodes(targets,targets_len,cfg.labels)
        preds.append(outputs)
        labels.append(lbls)
    preds = stt.flatten(preds)
    labels = stt.flatten(labels)
    metric = met_func(preds,labels)
    for i in range(3):
        print(f"{labels[i]} | {preds[i]}")
    return {
        "losses": losses,
        "metric": metric
    }