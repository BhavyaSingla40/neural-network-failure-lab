import numpy as np
from failure_lab.network import initialize_parameters, forward, backward, binary_cross_entropy, gradient_check, Optimizer
from failure_lab.train import metrics
from failure_lab.diagnostics import diagnose


def test_shapes_and_seed():
    p = initialize_parameters([31,16,8,1],seed=42)
    q = initialize_parameters([31,16,8,1],seed=42)
    assert p['W1'].shape == (31,16) and p['b1'].shape == (1,16)
    assert p['W2'].shape == (16,8) and p['W3'].shape == (8,1)
    assert all(np.array_equal(p[k],q[k]) for k in p)


def test_forward_backward_shapes():
    p=initialize_parameters([3,4,1]); X=np.ones((5,3)); y=np.array([0,1,0,1,0])
    z,c=forward(X,p)
    g,n=backward(y,p,c)
    assert z.shape == (5,1) and all(g[k].shape==p[k].shape for k in p)
    assert 'W1' in n


def test_stable_loss():
    y=np.array([0,1]); z=np.array([[-1000.],[1000.]])
    assert np.isfinite(binary_cross_entropy(z,y))
    assert binary_cross_entropy(z,y) < 1e-6


def test_gradient_check():
    rng=np.random.default_rng(42)
    X=rng.normal(size=(6,3)); y=np.array([0,1,1,0,0,1])
    p=initialize_parameters([3,5,1],init='xavier')
    r=gradient_check(X,y,p,activation='tanh',pos_weight=3,checks=40)
    assert r['max_relative_error'] < 1e-5, r


def test_optimizers_update():
    for method in ('sgd','adam'):
        p=initialize_parameters([3,2,1]); before=p['W1'].copy()
        Optimizer(method,lr=.01).step(p,{k:np.ones_like(v) for k,v in p.items()})
        assert not np.array_equal(p['W1'],before)


def test_metrics_and_diagnostic():
    m=metrics(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))
    assert m['pr_auc']==1 and m['recall']==1
    d=diagnose([],m,.5)
    assert d[0]['mode']=='none triggered'
