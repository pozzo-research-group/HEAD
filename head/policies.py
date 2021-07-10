import numpy as np
import pdb

def thomspon_sampling(optimizer, X,b=5):
	"""Batch Thompson Sampling
	Inputs:
	-------
		optimizer : a modAL.models.optimizer
		X  		  : design space points (array of shape (n_samples, dimension))
		b 		  : Batch size (int ;default, 5)  
	"""
	Xb = []
	fb = optimizer.estimator.sample_y(X,b, random_state=0)
	query_idx = []
	for fi in fb.T:
		max_id = np.argmax(fi)
		query_idx.append(max_id)
	query_idx = np.asarray(query_idx)    
	
	return query_idx, X[query_idx,:]
