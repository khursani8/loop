from utili.loops.multiclass import train_once,validate_once,get_preds
from utili import factory,utils
from torch.backends import cudnn
import numpy as np
import os
import torch
from pathlib import Path
import pandas as pd


args = utils.get_args()
cfg = utils.Config.fromfile(args.config)
cfg.version = args.version
cfg.fold = args.fold
cfg.working_dir = cfg.output_dir / cfg.version / str(cfg.fold)
Path(cfg.working_dir).mkdir(parents=True, exist_ok=True)
utils.fix_random_seeds(cfg.seed)
print("git:\n  {}\n".format(utils.get_sha()))

cudnn.benchmark = True

## get data
folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
dataset_train, loader_train = factory.get_dataset_loader(cfg.train, folds)
dataset_valid, loader_valid = factory.get_dataset_loader(cfg.valid, [cfg.fold])

## get model
model = factory.get_model(cfg.model)
device = factory.get_device(cfg.gpu)
model.cuda()
model.to(device)

## get optimizer
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
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    # fp16_scaler=fp16_scaler,
    # dino_loss=dino_loss,
)
start_epoch = to_restore["epoch"]

best_metric = to_restore["best_metric"]
# loop over the dataset multiple times
for epoch in range(start_epoch,cfg["epochs"]):
    trn_out = train_once(cfg,loader_train,model,optimizer,loss_func,scheduler)
    print(f'Trn Loss: {np.mean(trn_out["losses"][:10])} {np.mean(trn_out["losses"])} {np.mean(trn_out["losses"][-10:])}')

    val_out = validate_once(cfg,loader_valid,model,loss_func,metric_func)
    print(f'{epoch} Val Loss: {np.mean(val_out["losses"][:10])} {np.mean(val_out["losses"])} {np.mean(val_out["losses"][-10:])}')
    print(f'{cfg.metric.name}: {val_out["metric"]}')
    print("*"*24)

    if val_out["metric"] > best_metric:
        best_metric = val_out["metric"]
        save_dict = {
            "model":model.state_dict(),
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
    model=model
)

preds = get_preds(loader_valid,model)
sub = pd.read_csv(cfg.valid.data_path)
sub = sub[sub.fold.isin([cfg.fold])]
sub['pred'] = preds.argmax(-1)
score = metric_func(sub.pred.values, sub.target.values,None)
print(f'oof metric: {score:.4f}')
sub.to_csv(cfg.working_dir / f'oof_{cfg.version}_{score:.4f}.csv', index=False)
pd.DataFrame(preds).to_csv(cfg.working_dir / f'oof_{cfg.version}_{score:.4f}_raw.csv', index=False)
print(f'oof file was saved to: oof_{cfg.version}_{score:.4f}.csv')

dataset_test, loader_test = factory.get_dataset_loader(cfg.test, [0])
preds = get_preds(loader_test,model)
sub = pd.read_csv(cfg.test.data_path)
sub['target'] = preds.argmax(-1)
sub[['target']].to_csv(cfg.working_dir / f'sub_{cfg.version}.csv', index=False)
pd.DataFrame(preds).to_csv(cfg.working_dir / f'sub_{cfg.version}_raw.csv', index=False)