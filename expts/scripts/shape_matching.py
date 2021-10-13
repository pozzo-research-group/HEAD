from head import TestShapeMatchBO,SymmetricMatrices
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
import os, shutil, pickle

savedir = './figures/SphereCylinder/'
if  os.path.exists(savedir):
	shutil.rmtree(savedir)
os.makedirs(savedir)

def plot_target_proximities(obj, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
        
    train_x = obj.train_x.cpu().numpy()
    proximities = distance.cdist(train_x, obj.target.reshape(1,3))
    plot_scores = []
    for b in np.unique(obj.batch_number):
        scores = proximities[np.argwhere(obj.batch_number==b)]
        mu, std = scores.mean(), scores.std()
        plot_scores.append([mu, mu+std, mu-std])

    plot_scores = np.asarray(plot_scores)
    return ax.plot(np.unique(obj.batch_number), plot_scores[:,0])

def plot_best_trace(obj,ax):
    eval_nums = range(len(obj.train_obj))
    train_x = obj.train_x.cpu().numpy()
    proximities = distance.cdist(train_x, obj.target.reshape(1,3))
    best_trace = [proximities[:i].min() for i in eval_nums[1:]]
    best_trace.insert(0,proximities[0])
    return ax.plot(eval_nums, best_trace)
    
    
# define metrics
def ground_truth(xi,yi,xt,yt, pi, pt):
    rdd = np.sqrt(np.sum((pi[:2]-pt[:2])**2))
    spd = 0
    return -(rdd+spd)

def euclidean_dist(xi,yi,xt,yt, pi, pt):
    d = distance.euclidean(yi, yt)
    return -d

def sym_mat(xi,yi,xt,yt, pi, pt):
    M  = SymmetricMatrices(xt, yt,num_filters = 5)
    return -M.distance(yi)

def EMD(xi,yi,xt,yt, pi, pt):
    return -wasserstein_distance(yi, yt)
    
    
storage = {}

for epoch in range(10):
    expt = TestShapeMatchBO()
    fig, ax = plt.subplots()
    ax.axhline(0, ls='--', lw='2.0', color='k')
    logging.info('Epoch %d'%epoch)
    metrics = [ground_truth, euclidean_dist, sym_mat]
    for metric in metrics:
        logging.info('Metric :  %s'%metric.__name__)
        expt.run(metric)
        line = plot_best_trace(expt, ax)
        line[0].set_label(metric.__name__)
        storage[(epoch, metric.__name__)] = expt.save()
    ax.legend()
    ax.set_xlabel('Number of evaluations')
    ax.set_ylabel(r'$||X_{t} - X_{best}||_{2}$')        
    plt.savefig(savedir+'/epoch_%d.png'%epoch)
    with open(savedir + '/storage.pkl', 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
    plt.close()

