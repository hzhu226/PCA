import numpy as np
from scipy.linalg import eigh
from scipy.io import loadmat
import matplotlib.pyplot as mat

def load_and_center_dataset(filename):
    dataset = loadmat(filename)
    x = dataset['fea']
    center = x - np.mean(x, axis=0)
    return center

def get_covariance(dataset):
    setdata = np.transpose(dataset)
    covariance = np.dot(setdata, dataset)
    return covariance

def get_eig(S, m):
    length = len(S)
    value, v = eigh(S, eigvals=(length - m, length - 1))
    vec = np.fliplr(v)

    value_final = np.diag(np.flip(value))

    return value_final, vec

def project_image(image, U):
    project = np.zeros(shape=(len(image),))
    for x in range(len(U[0])):
        vec = U[:, x]
        y = np.dot(vec, image)
        project += np.dot(y, vec)
    return project

def display_image(orig, proj):
    tr_orig = np.transpose(np.reshape(orig, (32, 32)))
    tr_proj = np.transpose(np.reshape(proj, (32, 32)))
    ob, (ax_orig, ax_proj) = mat.subplots(1, 2)

    ax_orig.set_title('Original')
    ax_proj.set_title('Projection')

    image_orig = ax_orig.imshow(tr_orig, aspect='equal')
    image_proj = ax_proj.imshow(tr_proj, aspect='equal')
    ob.colorbar(image_orig, ax=ax_orig, fraction=0.046, pad=0.04)
    ob.colorbar(image_proj, ax=ax_proj, fraction=0.046, pad=0.04)
    mat.show()

