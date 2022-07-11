import torch

class Snapshot:
    def __init__(self, save_best_only=True, mode='min', initial_metric=None, output_dir='', name='', monitor='metric'):
        self.save_best_only = save_best_only
        if mode=='min':
            self.mode = 1
        elif mode=='max':
            self.mode = -1
        self.init_metric(initial_metric)
        self.output_dir = output_dir
        self.name = name
        self.monitor = monitor

    def init_metric(self, initial_metric):
        if initial_metric:
            self.best_metric = initial_metric
        else:
            self.best_metric = self.mode*1000

    def update_best_metric(self, metric):
        if self.mode*metric <= self.mode*self.best_metric:
            self.best_metric = metric
            return 1
        else:
            return 0

    def save_weight_optimizer(self, model, optimizer, prefix):
        torch.save(model.state_dict(), self.output_dir / str(self.name+f'_{prefix}.pt'))
        #torch.save(optimizer.state_dict(), self.output_dir / str(self.name+f'_optimizer_{prefix}.pt'))

    def snapshot(self, metric, model, optimizer, epoch):
        is_updated = self.update_best_metric(metric)
        if is_updated:
            self.save_weight_optimizer(model, optimizer, 'best')
            print('--> [best score was updated] save shapshot.')
        if not self.save_best_only:
            self.save_weight_optimizer(model, optimizer, f'epoch{epoch}')
            print(f'--> [epoch:{epoch}] save shapshot.')