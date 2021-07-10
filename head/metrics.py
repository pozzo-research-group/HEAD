from skfda.misc.metrics import LpDistance
from scipy.spatial import distance

def func_norm(query, target):
    d = LpDistance(p=2)
    val = -d(query.fd, target.fd)
    return float(val)

def euclidean_dist(query, target):
    val = -distance.euclidean(query.fd.data_matrix, target.fd.data_matrix)
    return float(val)

def oracle(query, st, measure):
    pi = get_length_radius(query)
    si = head.get_spectrum(pi[0],pi[1])
    yi = measure(si, st)

    return yi