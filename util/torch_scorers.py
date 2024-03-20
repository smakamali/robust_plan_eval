import torch

def subOpt(y_true, y_pred, opt_choice):
    trueOptimalET = torch.min(y_true)
    MLET = y_true[torch.argmin(y_pred)]
    if opt_choice.sum() == 1:
        db2_et = torch.squeeze(y_true[opt_choice == 1])
    else:
        db2_et = y_true[torch.argmin(y_pred)]
    MLsubOpt = MLET/trueOptimalET
    db2_subopt = db2_et/trueOptimalET
    return MLsubOpt, db2_subopt

def subOpt2(y_true, y_pred, opt_cost):
    trueOptimalET = torch.min(y_true)
    MLET = y_true[torch.argmin(y_pred)]
    db2_et = y_true[torch.argmin(opt_cost)]
    MLsubOpt = MLET/trueOptimalET
    db2_subopt = db2_et/trueOptimalET
    return MLsubOpt, db2_subopt

def q_error(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    eps = 0.000001
    denominator = torch.minimum(y_pred,y_true)
    numerator = torch.maximum(y_pred,y_true)
    qerror = (numerator+eps)/(denominator+eps)
    return qerror

def q_error_alt(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    eps = 0.000001
    denominator = y_pred
    numerator = y_true
    qerror = (numerator+eps)/(denominator+eps)
    return qerror