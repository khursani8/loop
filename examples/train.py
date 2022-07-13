from utili.loops.multiclass import train_once,validate_once,get_preds
from utili import factory,utils,optimizers
from torch.backends import cudnn
import numpy as np
import os
import torch
from pathlib import Path
import pandas as pd

from utili.utils.ema import EMA

cudnn.benchmark = False
args = utils.get_args()
cfg = utils.Config.fromfile(args.config)
cfg.version = args.version
cfg.fold = args.fold
cfg.working_dir = cfg.output_dir / cfg.version / str(cfg.fold)
Path(cfg.working_dir).mkdir(parents=True, exist_ok=True)
utils.fix_random_seeds(cfg.seed)
print("git:\n  {}\n".format(utils.get_sha()))


## get data
folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
dataset_train, loader_train = factory.get_dataset_loader(cfg.train, folds)
dataset_valid, loader_valid = factory.get_dataset_loader(cfg.valid, [cfg.fold])

## get model
model = factory.get_model(cfg.model)
device = factory.get_device(cfg.gpu)
model.cuda()
model.to(device)

ema = EMA(
    model,
    beta = cfg.ema.beta,              # exponential moving average factor
    update_after_step = 100,    # only after this number of .update() calls will it start updating
    update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
)

## get optimizer
if cfg.sam_optimizer:
    print(f'optimizer: SAM and {cfg.optim.name}')
    plist = [{'params': model.parameters(), 'lr': cfg.optim.lr, 'weight_decay': cfg.optim.weight_decay}]
    #base_optimizer = factory.get_optimizer(cfg.optim)(plist)
    base_optimizer = torch.optim.AdamW
    optimizer = optimizers.SAM(model.parameters(), base_optimizer, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
else:
    print(f'optimizer: {cfg.optim.name}')
    plist = [{'params': model.parameters(), 'lr': cfg.optim.lr, 'weight_decay': cfg.optim.weight_decay}]
    optimizer = factory.get_optimizer(cfg.optim)(plist)

## get loss
loss_func = factory.get_loss(cfg.loss)

## get metric
metric_func = factory.get_metric(cfg.metric)

# get scheduler.
scheduler = factory.get_scheduler(cfg.scheduler, optimizer)

# ============ optionally resume training ... ============
to_restore = {"epoch": 0,"best_metric":0}
utils.restart_from_checkpoint(
    os.path.join(cfg.working_dir, "checkpoint.pth"),
    run_variables=to_restore,
    ema=ema,
    optimizer=optimizer,
    scheduler=scheduler,
)
model.load_state_dict(ema.online_model.state_dict())
start_epoch = to_restore["epoch"]

best_metric = to_restore["best_metric"]
#########################################
cfg.iter = 0
for params in model.parameters():
    params.requires_grad = False
modules = model.model._modules
if "fc" in modules:
    pp = model.model.fc.parameters()
if "head" in modules:
    pp = model.model.head.parameters()
if "classifier" in modules:
    pp = model.model.classifier.parameters()
for p in pp:
    p.requires_grad = True
##########################################
for epoch in range(start_epoch,cfg["epochs"]):
    cfg.epoch = epoch
    trn_out = train_once(cfg,loader_train,model,optimizer,loss_func,scheduler,ema)
    print(f'Trn Loss: {np.mean(trn_out["losses"][:10])} {np.mean(trn_out["losses"])} {np.mean(trn_out["losses"][-10:])}')

    emaval_out = validate_once(cfg,loader_valid,ema,loss_func,metric_func)
    print(f'{epoch} emaVal Loss: {np.mean(emaval_out["losses"][:10])} {np.mean(emaval_out["losses"])} {np.mean(emaval_out["losses"][-10:])}')
    print(f'{cfg.metric.name}: {emaval_out["metric"]}')
    print("*"*24)

    if emaval_out["metric"] >= best_metric:
        best_metric = emaval_out["metric"]
        save_dict = {
            "ema": ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),
            "epoch":epoch + 1,
            "best_metric":best_metric
        }
        torch.save(save_dict,os.path.join(cfg.working_dir, 'checkpoint.pth'))

## predict oof
utils.restart_from_checkpoint(
    os.path.join(cfg.working_dir, "checkpoint.pth"),
    run_variables=to_restore,
    ema=ema
)

# calculate oof
preds = get_preds(loader_valid,ema)
sub = pd.read_csv(cfg.valid.data_path)
sub = sub[sub.fold.isin([cfg.fold])]
sub['pred'] = preds.argmax(-1)
score = metric_func(sub.pred.values, sub.target.values,None)
print(f'oof metric: {score:.4f}')
sub.to_csv(cfg.working_dir / f'oof_{cfg.version}_{score:.4f}.csv', index=False)
pd.DataFrame(preds).to_csv(cfg.working_dir / f'oof_{cfg.version}_{score:.4f}_raw.csv', index=False)
print(f'oof file was saved to: oof_{cfg.version}_{score:.4f}.csv')

# predict test set
dataset_test, loader_test = factory.get_dataset_loader(cfg.test, [0])
preds = get_preds(loader_test,ema)
sub = pd.read_csv(cfg.test.data_path)
sub['target'] = preds.argmax(-1)
sub[['target']].to_csv(cfg.working_dir / f'sub_{cfg.version}.csv', index=False)
pd.DataFrame(preds).to_csv(cfg.working_dir / f'sub_{cfg.version}_raw.csv', index=False)