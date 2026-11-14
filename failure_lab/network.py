"""Pure NumPy fully-connected binary classifier; each row is a sample."""
import numpy as np


def sigmoid(x):
    # Clip only the exponent's input to avoid overflow; loss uses raw logits.
    x = np.asarray(x)
    return np.where(x >= 0, 1 / (1 + np.exp(-np.clip(x, -500, 500))),
                    np.exp(np.clip(x, -500, 500)) / (1 + np.exp(np.clip(x, -500, 500))))


def initialize_parameters(layer_dims, init='he', seed=42):
    if len(layer_dims) < 2 or any(not isinstance(d, (int, np.integer)) or d <= 0 for d in layer_dims):
        raise ValueError('layer_dims must contain at least two positive integers')
    rng = np.random.default_rng(seed)
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:]), 1):
        if init == 'he': scale = np.sqrt(2 / fan_in)
        elif init == 'xavier': scale = np.sqrt(1 / fan_in)
        elif init == 'zero': scale = 0
        else: raise ValueError('init must be he, xavier or zero')
        params[f'W{i}'] = rng.standard_normal((fan_in, fan_out)) * scale
        params[f'b{i}'] = np.zeros((1, fan_out))
        assert params[f'W{i}'].shape == (fan_in, fan_out)
    return params


def forward(X, params, activation='relu'):
    if activation not in ('relu', 'sigmoid', 'tanh', 'leaky_relu'):
        raise ValueError('Unknown activation')
    a = np.asarray(X, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != params['W1'].shape[0]:
        raise ValueError('Input must be a 2D matrix with matching feature width')
    caches = []
    L = len(params) // 2
    for l in range(1, L):
        z = a @ params[f'W{l}'] + params[f'b{l}']
        if activation == 'relu': next_a = np.maximum(z, 0)
        elif activation == 'leaky_relu': next_a = np.where(z > 0, z, .01*z)
        elif activation == 'sigmoid': next_a = sigmoid(z)
        else: next_a = np.tanh(z)
        caches.append((a, z, next_a))
        a = next_a
    logits = a @ params[f'W{L}'] + params[f'b{L}']
    caches.append((a, logits, sigmoid(logits)))
    return logits, caches


def binary_cross_entropy(logits, y, pos_weight=1.0):
    """Weighted mean BCE, stable on extreme logits; y has shape (batch,1)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if logits.shape != y.shape: raise ValueError('Logit and label shapes differ')
    weights = 1 + (pos_weight - 1) * y
    # softplus(z) - y*z, computed without overflow
    loss = np.maximum(logits, 0) - logits*y + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(weights * loss))


def backward(y, params, caches, activation='relu', pos_weight=1.0):
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    m = len(y)
    if len(caches[-1][2]) != m: raise ValueError('Batch length mismatch')
    grads, grad_norms = {}, {}
    L = len(caches)
    weights = 1 + (pos_weight - 1) * y
    dz = (caches[-1][2] - y) * weights / m
    for l in range(L, 0, -1):
        a_prev, _, _ = caches[l-1]
        grads[f'W{l}'] = a_prev.T @ dz
        grads[f'b{l}'] = np.sum(dz, axis=0, keepdims=True)
        grad_norms[f'W{l}'] = float(np.linalg.norm(grads[f'W{l}']))
        if l > 1:
            da = dz @ params[f'W{l}'].T
            z = caches[l-2][1]
            if activation == 'relu': dz = da * (z > 0)
            elif activation == 'leaky_relu': dz = da * np.where(z > 0, 1, .01)
            elif activation == 'sigmoid':
                s = caches[l-2][2]
                dz = da * s * (1-s)
            elif activation == 'tanh': dz = da * (1 - caches[l-2][2]**2)
            else: raise ValueError('Unknown activation')
    return grads, grad_norms


class Optimizer:
    def __init__(self, method='adam', lr=.001):
        if method not in ('adam', 'sgd') or lr <= 0: raise ValueError('Invalid optimizer / learning rate')
        self.method, self.lr = method, lr
        self.m, self.v, self.step_count = {}, {}, 0

    def step(self, params, grads):
        self.step_count += 1
        for name in params:
            g = grads[name]
            if not np.all(np.isfinite(g)): raise FloatingPointError(f'Nonfinite gradient: {name}')
            if self.method == 'sgd': params[name] -= self.lr * g
            else:
                self.m[name] = .9*self.m.get(name, np.zeros_like(g)) + .1*g
                self.v[name] = .999*self.v.get(name, np.zeros_like(g)) + .001*(g*g)
                mhat = self.m[name] / (1-.9**self.step_count)
                vhat = self.v[name] / (1-.999**self.step_count)
                params[name] -= self.lr*mhat/(np.sqrt(vhat)+1e-8)


def predict_proba(X, params, activation='relu', batch_size=8192):
    return np.concatenate([sigmoid(forward(X[i:i+batch_size], params, activation)[0]).ravel()
                           for i in range(0, len(X), batch_size)])


def gradient_check(X, y, params, activation='tanh', pos_weight=1.0, eps=1e-5, checks=25, seed=42):
    """Central finite differences over randomly sampled scalar parameters."""
    rng = np.random.default_rng(seed)
    _, caches = forward(X, params, activation)
    grads, _ = backward(y, params, caches, activation, pos_weight)
    errors = []
    names = list(params)
    for _ in range(checks):
        name = names[int(rng.integers(len(names)))]
        p = params[name]
        index = tuple(int(rng.integers(dim)) for dim in p.shape)
        orig = p[index]
        try:
            p[index] = orig + eps
            plus = binary_cross_entropy(forward(X, params, activation)[0], y, pos_weight)
            p[index] = orig - eps
            minus = binary_cross_entropy(forward(X, params, activation)[0], y, pos_weight)
        finally:
            p[index] = orig
        numerical = (plus-minus)/(2*eps)
        analytical = grads[name][index]
        errors.append(abs(numerical-analytical)/max(1e-7, abs(numerical)+abs(analytical)))
    return {'max_relative_error': float(max(errors)), 'mean_relative_error': float(np.mean(errors)), 'checks': checks}
