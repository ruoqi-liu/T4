from scipy.special import softmax
import numpy as np
from tqdm import tqdm
import random
import pickle


def simulate_mimic(data_dir, dataset_name, output):
    data = pickle.load(open(data_dir + dataset_name + '.pkl', 'rb'))
    features = pickle.load(open(data_dir + 'header.pkl', 'rb'))
    demos = ['agegroup', 'heightgroup', 'weightgroup', 'gender']

    n_features = len(features) + len(demos)
    s = np.random.multivariate_normal(np.zeros(shape=(n_features)), 0.1 * np.eye(n_features))
    m = np.random.normal(0, 0.1)

    cov = np.random.uniform(low=-1, high=1, size=(n_features + 1, n_features + 1))
    w_cov = 0.1 * (np.dot(cov, cov.T))
    mu1 = np.zeros(shape=(n_features + 1))
    mu0 = np.ones(shape=(n_features + 1)) * 0.1
    w1 = np.random.multivariate_normal(mean=mu1, cov=w_cov, size=(1,)).T
    w0 = np.random.multivariate_normal(mean=mu0, cov=w_cov, size=(1,)).T

    data_sim = data.copy()

    for id, datum in tqdm(data.items()):
        x = []
        for val in features:
            x.append(datum[val])
        x = np.array(x).T

        x_demo = []
        for demo in demos:
            x_demo.append(datum[demo])
        x_demo = np.array(x_demo)

        x_all, t, y_f, y_cf = [], [], [], []
        timestamps = len(x)
        for ts in range(timestamps):
            x_cur = np.concatenate((x[ts], x_demo))

            if ts > 0:
                p = min(ts + 1, 6)
                weights = np.arange(1, p)
                weights = softmax(weights)
                x_avg = np.average(np.array(x_all)[-(p-1):], weights=weights, axis=0)
            else:
                x_avg = x_cur

            x_all.append(x_cur)

            t_cur = np.matmul(x_avg, s) + m
            t_cur = np.random.binomial(1, 1 / (1 + np.exp(-t_cur)))
            t.append(t_cur)

            tmp = np.concatenate((x_avg, [t_cur]))
            y_t_f = t_cur * np.matmul(tmp, w1) + (1 - t_cur) * np.matmul(tmp,w0)
            y_t_cf = t_cur * np.matmul(tmp, w0) + (1 - t_cur) * np.matmul(tmp,w1)

            y_f.append(y_t_f)
            y_cf.append(y_t_cf)

        data_sim[id]['treatment'] = t
        data_sim[id]['outcome'] = [np.concatenate(y_f), np.concatenate(y_cf)]

    pickle.dump(data_sim, open(f'{output}', 'wb'))
    print(f'Save synthetic-MIMIC data in {output}')

if __name__ == '__main__':
    simulate_mimic('data/', 'mimic3', 'data/synthetic_mimic3.pkl')


