"""Use exactly once after tuning; never use held-out results for model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from failure_lab.data import load_data
from failure_lab.network import predict_proba
from failure_lab.train import metrics

p=argparse.ArgumentParser()
p.add_argument('--experiment',default='healthy')
p.add_argument('--csv',default='creditcard.csv')
p.add_argument('--threshold',type=float,default=.5)
p.add_argument('--confirm-final',action='store_true')
a=p.parse_args()
if not a.confirm_final: p.error('Explicitly pass --confirm-final after selecting the final model on validation.')
if not (0 < a.threshold < 1): p.error('threshold must be between 0 and 1')
path=Path('results') / f'{a.experiment}.json'
record=json.loads(path.read_text())
if 'model_file' not in record: p.error('This experiment has no saved DNN')
(_,_,Xt),(_,_,yt),cols=load_data(a.csv,record['config']['seed'])
if cols != record['features']: raise ValueError('Feature order mismatch')
with np.load(record['model_file']) as saved: params={k:saved[k] for k in saved.files}
score=metrics(yt,predict_proba(Xt,params,record['config']['activation']),a.threshold)
output={'experiment':a.experiment,'threshold':a.threshold,'test':score,
        'warning':'Final held-out test. Do not tune model or threshold using this output.'}
Path('results/final_test.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
