from optimizers.basic_optimizer import get_basic_adam, get_basic_adamw, get_basic_sgd, get_ranger_ada_belief
from configs import Config


def get_optimizer(model, cfg: Config):
    if cfg.optimizer.name == 'Adam':
        return get_basic_adam(model, cfg.optimizer.param)
    elif cfg.optimizer.name == 'AdamW':
        return get_basic_adamw(model, cfg.optimizer.param)
    elif cfg.optimizer.name == 'SGD':
        return get_basic_sgd(model, cfg.optimizer.param)
    elif cfg.optimizer.name == 'RangerAdaBelief':
        return get_ranger_ada_belief(model, cfg.optimizer.param)

