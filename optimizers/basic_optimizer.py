from torch.optim import Adam, SGD, AdamW
# from ranger_adabelief import RangerAdaBelief


def get_basic_adam(model, params):
    return Adam(model.parameters(), **params)


def get_basic_adamw(model, params):
    return AdamW(model.parameters(), **params)


def get_basic_sgd(model, params):
    return SGD(model.parameters(), **params)


def get_ranger_ada_belief(model, params):
    return None
    # return RangerAdaBelief(model.parameters(), **params)


