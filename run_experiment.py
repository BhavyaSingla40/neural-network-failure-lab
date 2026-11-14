"""Run one experiment without ever evaluating on the held-out test set."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from failure_lab.data import load_data
from failure_lab.network import initialize_parameters, gradient_check
from failure_lab.train import fit, metrics
from failure_lab.diagnostics import diagnose

CONFIGS = {
    'healthy': dict(layer_dims=[31,16,8,1], activation='relu', init='he', optimizer='adam', lr=.001, batch_size=512, epochs=10, pos_weight=1.),
    'vanishing': dict(layer_dims=[31,16,16,16,16,8,1], activation='sigmoid', init='xavier', optimizer='sgd', lr=.001, batch_size=512, epochs=8, pos_weight=1.),
    'overfit': dict(layer_dims=[31,128,64,1], activation='relu', init='he', optimizer='adam', lr=.003, batch_size=128, epochs=30, pos_weight=1.),
    'imbalance': dict(layer_dims=[31,16,8,1], activation='relu', init='he', optimizer='adam', lr=.001, batch_size=512, epochs=10, pos_weight=1.),
    'imbalance_fix': dict(layer_dims=[31,16,8,1], activation='relu', init='he', optimizer='adam', lr=.001, batch_size=512, epochs=10, pos_weight=200.),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='creditcard.csv')
    parser.add_argument('--experiment', choices=['baseline','gradient_check',*CONFIGS], default='healthy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    (Xtr,Xv,_), (ytr,yv,_), columns = load_data(args.csv,args.seed)
    # Test split deliberately discarded here. Only final_evaluate.py loads it for scoring.
    out = Path('results'); out.mkdir(exist_ok=True)
    if args.experiment == 'baseline':
        clf = LogisticRegression(class_weight='balanced', random_state=args.seed, max_iter=1000)
        clf.fit(Xtr,ytr)
        score = metrics(yv, clf.predict_proba(Xv)[:,1])
        result = {'experiment':'baseline','config':{'model':'LogisticRegression', 'class_weight':'balanced', 'seed':args.seed}, 'validation':score}
    elif args.experiment == 'gradient_check':
        rng=np.random.default_rng(args.seed)
        p=initialize_parameters([3,4,1],init='xavier',seed=args.seed)
        X=rng.normal(size=(6,3)); y=np.array([0,1,0,1,0,1])
        result={'experiment':'gradient_check','result':gradient_check(X,y,p,'tanh',pos_weight=2)}
    else:
        cfg = dict(CONFIGS[args.experiment]); cfg['seed'] = args.seed
        if args.epochs is not None: cfg['epochs'] = args.epochs
        # Only overfit uses a small deterministic, stratified subset to accentuate variance.
        if args.experiment == 'overfit':
            from sklearn.model_selection import train_test_split
            ids, _ = train_test_split(np.arange(len(ytr)), train_size=1200, random_state=args.seed, stratify=ytr)
            Xtr,ytr=Xtr[ids],ytr[ids]
        params, history, score, best_epoch = fit(Xtr,ytr,Xv,yv,cfg)
        notes = diagnose(history,score,float(np.mean(ytr)))
        model_path = out / f'{args.experiment}_model.npz'
        np.savez_compressed(model_path,**params)
        result = {'experiment':args.experiment,'config':cfg,'features':columns,
                  'validation':score,'history':history,'diagnostics':notes,'best_epoch':best_epoch,
                  'model_file':str(model_path), 'created_utc':datetime.now(timezone.utc).isoformat()}
    path=out / f'{args.experiment}.json'; path.write_text(json.dumps(result,indent=2))
    print('Saved',path)
    print(json.dumps(result.get('validation',result.get('result')),indent=2))

if __name__ == '__main__': main()
