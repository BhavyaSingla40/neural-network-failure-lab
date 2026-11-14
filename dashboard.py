"""Read-only Streamlit dashboard for logged experiments (no training inside UI)."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
st.set_page_config(page_title='Neural Network Failure Lab',layout='wide')
st.title('Neural Network Failure Lab')
st.caption('Build → Break → Observe → Diagnose → Fix → Explain. Static recorded results only.')
paths=sorted(p for p in Path('results').glob('*.json') if p.name != 'final_test.json')
records=[]
for p in paths:
    try: records.append(json.loads(p.read_text()))
    except (ValueError,OSError): st.warning(f'Could not read {p.name}')
if not records:
    st.info('No experiment logs found. Run: python run_experiment.py --experiment healthy')
    st.stop()
rows=[]
for r in records:
    m=r.get('validation',{})
    if m: rows.append({'experiment':r['experiment'], **{k:m.get(k) for k in ['pr_auc','precision','recall','f1','roc_auc']}})
st.subheader('Validation metric comparison')
if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
options=[r['experiment'] for r in records]
name=st.selectbox('Inspect an experiment',options)
r=next(r for r in records if r['experiment']==name)
if 'config' in r: st.json(r['config'])
if 'validation' in r:
    st.subheader('Validation confusion matrix (TN, FP / FN, TP)')
    st.table(pd.DataFrame(r['validation']['confusion_matrix'],columns=['Pred 0','Pred 1'],index=['True 0','True 1']))
if 'history' in r:
    df=pd.DataFrame(r['history']).set_index('epoch')
    a,b=st.columns(2)
    with a: st.subheader('Loss'); st.line_chart(df[['train_loss','val_loss']])
    with b: st.subheader('PR-AUC'); st.line_chart(df[['train_pr_auc','val_pr_auc']])
    st.subheader('Gradient norms')
    st.line_chart(df[['grad_W1','grad_Wlast']])
    st.subheader('Rule-based diagnostics')
    for msg in r['diagnostics']:
        st.markdown(f"**{msg['mode']}** — {msg['evidence']}  ")
        st.write('Suggested next step:',msg['fix'])
st.caption('ROC-AUC can look strong on rare-fraud data even when precision is poor. Prioritize PR-AUC and the confusion matrix.')
