# a basic configuration unit

from ray import tune

config = {
    'optimizer.params.lr': tune.grid_search(
            [0.0003, 0.0006]
        ),
    'path': 'loss_0707/bce.yaml',
    'train.batch_size': tune.grid_search(
            [64, 128]
        ),
    'basic.amp': 'O1'
}
