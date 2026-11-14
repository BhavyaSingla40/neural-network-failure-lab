"""Training utilities with validation-only checkpoint selection."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .network import initialize_parameters, forward, backward, binary_cross_entropy, predict_proba, Optimizer


def metrics(y, prob, threshold=.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {'pr_auc': float(average_precision_score(y, prob)),
            'roc_auc': float(roc_auc_score(y, prob)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0,1]).tolist()}


def fit(X_train, y_train, X_val, y_val, config):
    rng = np.random.default_rng(config['seed'])
    params = initialize_parameters(config['layer_dims'], config['init'], config['seed'])
    opt = Optimizer(config['optimizer'], config['lr'])
    history = []
    best_score, best_params, best_epoch = -float('inf'), None, None
    batch_size = config['batch_size']
    for epoch in range(1, config['epochs']+1):
        order = rng.permutation(len(y_train))
        epoch_grad = {}
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            logits, cache = forward(X_train[ids], params, config['activation'])
            grads, norms = backward(y_train[ids], params, cache, config['activation'], config['pos_weight'])
            opt.step(params, grads)
            epoch_grad = norms
        train_prob = predict_proba(X_train, params, config['activation'])
        val_prob = predict_proba(X_val, params, config['activation'])
        # Stable logit losses across batches, no test access.
        def eval_loss(X, y):
            total = 0.
            for i in range(0, len(y), 8192):
                end = i+8192
                logits, _ = forward(X[i:end], params, config['activation'])
                total += binary_cross_entropy(logits, y[i:end], config['pos_weight']) * len(y[i:end])
            return total / len(y)
        train_loss = eval_loss(X_train, y_train)
        val_loss = eval_loss(X_val, y_val)
        val_m = metrics(y_val, val_prob)
        last_key = f"W{len(config['layer_dims'])-1}"
        history.append({'epoch':epoch, 'train_loss':train_loss, 'val_loss':val_loss,
                        'train_pr_auc':float(average_precision_score(y_train,train_prob)),
                        'val_pr_auc':val_m['pr_auc'], 'grad_W1':epoch_grad.get('W1',0.),
                        'grad_Wlast':epoch_grad.get(last_key,0.)})
        print(f"epoch={epoch:02d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_PR_AUC={val_m['pr_auc']:.5f}", flush=True)
        # Model selection by validation PR-AUC only.
        if val_m['pr_auc'] > best_score:
            best_score, best_epoch = val_m['pr_auc'], epoch
            best_params = {name: val.copy() for name,val in params.items()}
    val_prob = predict_proba(X_val, best_params, config['activation'])
    return best_params, history, metrics(y_val,val_prob), best_epoch
