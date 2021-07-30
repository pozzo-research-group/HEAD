from scipy.spatial import distance

def func_norm(query, target):
    from skfda.misc.metrics import LpDistance

    """L2-distance between two functions
    inputs:
    ------
        query, target : query and target spectra as a head.Spectra1D object

    outputs:
    --------
        distance as a float

    """

    d = LpDistance(p=2)
    val = -d(query.fd, target.fd)
    return float(val)

def euclidean_dist(query, target):
    """Euclidean distance between two vectors
    Caution: This function DO NOT consider the spectra as functions but as list of numebrs
    inputs:
    ------
        query, target : query and target spectra as numpy arrays of same shape

    outputs:
    --------
        distance as a float

    """
    val = -distance.euclidean(query, target)
    return float(val)