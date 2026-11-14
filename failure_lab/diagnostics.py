"""Transparent heuristic warnings, not claims of proven causation."""

def diagnose(history, val_metrics, train_fraud_ratio, threshold=0.5):
    messages = []
    if len(history) >= 3:
        recent = history[-3:]
        first = [r.get('grad_W1', float('nan')) for r in recent]
        last = [r.get('grad_Wlast', float('nan')) for r in recent]
        if all(x >= 0 for x in first+last) and min(last) > 1e-7 and max(first) < 1e-3 * min(last):
            messages.append({'mode':'vanishing gradients', 'evidence':'First-layer gradient norms much smaller than last-layer norms for three epochs', 'fix':'Try ReLU and He initialization; compare norm trajectories'})
        if all(r['val_loss'] > r['train_loss'] * 1.5 for r in recent) and recent[-1]['val_loss'] > recent[0]['val_loss']:
            messages.append({'mode':'overfitting', 'evidence':'Validation loss rising and >1.5x training loss', 'fix':'Reduce capacity, add early stopping, or more data'})
    if val_metrics['recall'] < .2 and val_metrics['pr_auc'] <= max(.05, 3 * train_fraud_ratio):
        messages.append({'mode':'class imbalance', 'evidence':f"Low recall at threshold {threshold} and PR-AUC near low baseline", 'fix':'Try positive class weighting, then tune threshold using validation only'})
    return messages or [{'mode':'none triggered', 'evidence':'No defined heuristic crossed its threshold', 'fix':'Inspect plots and compare controlled runs'}]
