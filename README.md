# predictive-system-level-synthesis-master
### Hi there 👋, here is a brief ReadMe about our experiments for predictive system-level synthesis (SLS). This is still a work in progress.
## System Requirement
<!--
We recommend using Python 3 (and pip3) or above. 
-->
* Python 3 or higher
* Python pip3
* cvxpy 1.4.2
* numpy 1.26.2
* control 0.9.2

## Get Started
To get started, please run the file ``Predictive Ricatti Control.py`` and  ``Predictive SLS.py``.

The former represents the causal control and predictive control under the Ricatti method. The latter represents the causal control and predictive control under the SLS.

## Mathematical Interpretation
* Parameters:
  - horizon: Length of a finite horizon.
  - (A, B1, B2, C1, D12): System matrices, in state-feedback lqc control, we have (A, B1, B2, C1, D12) = (A, I, B, Q, R).

* Objective Fucntion:
  - For SLS, a convex optimization problem we have implemented is as follows:
    > SLS = min_{Phi_x, Phi_u} || [[ Q^{1/2}, 0 ]; [ 0, R^{1/2} ]] [[ Phi_x ]; [ Phi_u ]] ||^2_Frob
  
  - For predictive SLS, a convex optimization problem we have implemented is as follows:
    > PSLS = min_{Phi_x, Phi_u, Phi_hat_x, Phi_hat_u} || [[Q^{1/2}, 0]; [0, R^{1/2}]] [[Phi_x + Phi_hat_x]; [Phi_u + Phi_hat_u]] ||^2_Frob
  
* Contraints:
   - For SLS,  state-feedback constraints is:
     > In theory [ zI-A, -B2 ][[ Phi_x ]; [ Phi_u ]] = I
     
     > Implementation Phi_x[t + 1] = A * Phi_x[t] + B2 * Phi_u[t]

   - For Predictive SLS, state-feedback constriants is:
     > [ zI-A, -B2 ][[ Phi_x Phi_hat_x ]; [ Phi_u Phi_hat_u ]] = [ I, 0 ]
     
     > Implementation [ Phi_x[t + 1], Phi_hat_x[t - 1] ] = A * [ Phi_x[t], Phi_hat_x[t] ] + B2 * [ Phi_u[t], Phi_hat_u[t] ]
## Current Progress
For testing, we output K and L_hat and the synthesized counterparts in ``controller_models.py`` and ``algorithms.py`` respectively, and we find intuitive consistency between SLS and Riccati's method in our tracking experiments.
