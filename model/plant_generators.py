from .system_models import LTI_System
from math import floor, ceil
import numpy as np

'''
Some helper functions to generate the LTI system plant matrices
'''
def generate_matrices_from_ABCD (system_model=None, A=None, B=None, C=None, D=None):
    '''
    Given A, B, C, and D, partition them into the corresponding matrices
    '''

    n_x = system_model._n_x
    Ny = system_model._n_y
    Nz = system_model._n_z
    n_u = system_model._n_u
    n_w = system_model._n_w

    system_model._A = A
    if B is None:
        system_model._B1 = None
        system_model._B2 = None
    else:
        system_model._B1 = B[:,0:n_w]
        system_model._B2 = B[:,n_w:n_w+n_u]

    if C is None:
        system_model._C1 = None
        system_model._C2 = None
    else:
        system_model._C1 = C[0:Nz,:]
        system_model._C2 = C[Nz:Nz+Ny,:]
    
    if D is None:
        system_model._D11 = None
        system_model._D12 = None
        system_model._D21 = None
        system_model._D22 = None
    else:
        system_model._D11 = D[0:Nz,0:n_w]
        system_model._D12 = D[0:Nz,n_w:n_w+n_u]
        system_model._D21 = D[Nz:Nz+Ny,0:n_w]
        system_model._D22 = D[Nz:Nz+Ny,n_w:n_w+n_u]

def generate_BCD_and_zero_initialization (system_model=None):
    # This function simply serves as an abbreviation
    # The user has to ensure that system_model is LTI_System
    system_model._n_w = system_model._n_x
    system_model._Nz = system_model._n_x + system_model._n_u

    system_model._B1  = np.eye(system_model._n_x)
    system_model._C1  = np.eye(system_model._n_x)
    system_model._D12 = np.eye(system_model._n_u)

    if not system_model._state_feedback:
        # assign the matrices for y as well
        system_model._C2  = np.eye(system_model._n_y, system_model._n_x)
        #system_model._D22 = np.eye(system_model._Ny, system_model._n_u)

    system_model.initialize ()

def generate_doubly_stochastic_chain (system_model=None, rho=0, actuator_density=1, alpha=0):
    if not isinstance(system_model,LTI_System):
        # only modify LTI_System plant
        return
    
    if system_model._n_x == 0:
        return 

    n_x = system_model._n_x
    n_u = int(ceil(n_x*actuator_density))
    system_model._n_u = n_u

    system_model._A = (1-2*alpha)*np.eye(n_x)
    system_model._A[0,0] += alpha
    system_model._A[n_x-1,n_x-1] += alpha
    tmp = alpha*np.eye(n_x-1)
    system_model._A[0:-1,1:] += tmp
    system_model._A[1:,0:-1] += tmp
    system_model._A *= rho

    system_model._B2 = np.zeros([n_x,n_u])
    for i in range (n_u):
        x = int(floor(i/actuator_density)) % n_x
        system_model._B2[x,i] = 1

def generate_random_chain (system_model=None, rho=1, actuator_density=1, random_seed=None):
    if not isinstance(system_model,LTI_System):
        # only modify LTI_System plant
        return

    if system_model._n_x == 0:
        return

    if random_seed is not None:
        np.random.seed(seed=random_seed)

    n_x = system_model._n_x
    n_u = int(ceil(n_x*actuator_density))
    system_model._n_u = n_u

    system_model._A = np.eye(n_x)

    if n_x > 1:
        system_model._A[0:-1,1:] += np.diag(np.random.randn(n_x-1))
        system_model._A[1:,0:-1] += np.diag(np.random.randn(n_x-1))

    eigenvalues, eigenvectors = np.linalg.eig(system_model._A)
    largest_eigenvalue = np.max(np.absolute(eigenvalues))

    # normalization
    system_model._A /= largest_eigenvalue
    system_model._A *= rho

    system_model._B2 = np.zeros([n_x,n_u])
    for i in range (n_u):
        x = int(floor(i/actuator_density)) % n_x
        system_model._B2[x,i] = np.random.randn ()