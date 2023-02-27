import numpy as np
from scipy.special import softmax
import pickle

# parameters
N = 5000   # number of patients
T = 50 # number of timestamps
k = 25  # number of covariates
k_t = 20 # number of temporal covariates

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def gen_next(X, A ,t):

    lam = np.random.normal(0, 0.2, size=(N, 1))

    weights = np.arange(1, t+1)

    weights = softmax(weights)

    x_t = np.average(X[:, :, :t], axis=-1, weights=weights) +lam* np.average(A[:, :, :t], axis=-1, weights=weights)
    return x_t


def gen_factual_data():

    # initial state
    mu = np.zeros(shape=(k))
    cov = np.random.uniform(low=-1, high=1, size=(k,k))

    x_cov = 0.1*(np.dot(cov,cov.T))

    x = np.random.multivariate_normal(mean=mu, cov=x_cov, size=(N,))

    x_t = x[:, :k_t]
    x_sta = x[:, k_t:]

    m = np.random.normal(0, 0.1, (N,))
    s = np.random.multivariate_normal(mean=np.zeros(shape=(k)), cov=0.1 * np.identity(k), size=(1,)).T

    prob = np.matmul(x, s).squeeze(-1) + m

    prob = sigmoid(prob)
    a = np.random.binomial(n=1, p=prob, size=(N,))

    a = a.reshape((N,1))

    X = np.zeros(shape=(N, k_t, T))

    X[:,:,0] = x_t
    A = np.zeros(shape=(N, 1, T))
    A[:,:,0] = a

    # simulate y
    cov = np.random.uniform(low=-1, high=1, size=(k, k))
    w_cov = 0.1 * (np.dot(cov, cov.T))
    mu1 = np.zeros(shape=(k))
    mu0 = np.ones(shape=(k))
    w1 = np.random.multivariate_normal(mean=mu1, cov=w_cov, size=(1,)).T
    w0 = np.random.multivariate_normal(mean=mu0, cov=w_cov, size=(1,)).T
    Y = np.zeros(shape=(N, 2, T))

    tmp = np.concatenate((X[:, :, 0], x_sta), axis=1)
    y_f = a * np.matmul(tmp, w1) + (1 - a) * np.matmul(tmp, w0)
    y_cf = a * np.matmul(tmp, w0) + (1 - a) * np.matmul(tmp, w1)

    Y[:, 0, 0] = y_f.squeeze()
    Y[:, 1, 0] = y_cf.squeeze()

    for t in range(1, T):

        x_t = gen_next(X, A, t)
        X[:,:,t] = x_t

        prob = np.matmul(np.concatenate((x_t, x_sta), axis=1), s).squeeze(-1) + m
        prob = sigmoid(prob)

        a = np.random.binomial(n=1, p=prob, size=(N,))
        a = a.reshape((N, 1))
        A[:, :, t] = a

        tmp = np.concatenate((x_t,x_sta),axis=1)
        y_f = a * np.matmul(tmp, w1) + (1 - a) * np.matmul(tmp, w0)
        y_cf = a * np.matmul(tmp, w0) + (1 - a) * np.matmul(tmp, w1)

        Y[:, 0, t] = y_f.squeeze()
        Y[:, 1, t] = y_cf.squeeze()

    Y = (Y - np.mean(Y)) / np.std(Y)

    x_mean = np.mean(X, axis=(0, 2))
    x_std = np.std(X, axis=(0, 2))
    X_norm = np.zeros_like(X)
    for i in range(k_t):
        X_norm[:,i,:] = (X[:,i,:]-x_mean[i])/x_std[i]

    X_static_norm = np.zeros_like(x_sta)
    x_static_mean = np.mean(x_sta, axis=-1)
    x_static_std = np.std(x_sta, axis=-1)
    for i in range(k-k_t):
        X_static_norm[:,i] = (x_sta[:,i]-x_static_mean[i])/x_static_std[i]


    results = {'X': X_norm, 'A_factual': A, 'Y_factual': Y, 'X_sta': X_static_norm}


    return results


if __name__ == '__main__':

    data = gen_factual_data()

    X, A, Y, X_sta = data['X'], data['A_factual'], data['Y_factual'], data['X_sta']


    data_out = {}
    for i in range(N):
        x, x_sta = X[i], X_sta[i]
        a, y = A[i,0,:], Y[i,:,:]

        datum = {}
        datum['x'] = x
        datum['x_static'] = x_sta
        datum['y'] = y
        datum['a'] = a
        data_out[i] = datum

    data_name = 'synthetic_full'
    pickle.dump(data_out, open('data/{}.pkl'.format(data_name), 'wb'))




