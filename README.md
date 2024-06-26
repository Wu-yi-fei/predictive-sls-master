# predictive-system-level-synthesis-master
### Hi there 👋, here is a brief README about our experiments for predictive system-level synthesis (pSLS). This is still a work in progress.
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
* Parameters
  - horizon: Length of a finite horizon.
  - $(A, B_1, B_2, C_1, D_{12})$: System matrices, in state-feedback lqc control, we have $`(A, B_1, B_2, C_1, D_{12}) = (A, I, B, Q, R)`$.
 
* System
  - State feed-back system:
    > $x = A x + B u + w, y = x$
  - lqc:
    > $\mathrm{lqc} = x^* Q x + u^* R u$

* Controller
  - For **causal Riccati controller**, we set the policy as follows:
     > $u[t] = K x[t] = (R+B^* PB)^{-1}BPA x[t]$
     
  - For **predictive Riccati controller**, we set the policy as follows:
     > $u[t] = K x[t] + \sum_{t} \widehat{L} \widehat{w}[t] = (R+B^* PB)^{-1}(BPA x[t] + \sum_{t} (F^*)^t \widehat{w}[t])$

  - For **SLS controller**, we set the policy as follows:
     > $u[t] = \sum_{t} \Phi_u[t] w[t]$

  - For **predictive SLS controller**, we set the policy as follows:
     > $u[t] = \sum_{t} \Phi_u[t] w[t] + \sum_{t} \widehat{\Phi}_u[t] \widehat{w}[t]$

  - For **Babak Nocausal controller**, we set the policy as follows:
     > TODO
    
* Objective Fucntion:
  - For **SLS**, a convex optimization problem we have implemented is as follows:
    > $\mathbf{SLS} = \mathrm{min}_{\Phi_x, \Phi_u} || \left[Q^{1/2}, 0; 0, R^{1/2} \right] [\Phi_x; \Phi_u] ||^2_F$
  
  - For **predictive SLS**, a convex optimization problem we have implemented is as follows:
    > $\mathbf{pSLS} = \mathrm{min}_{\Phi_x, \Phi_u, \widehat{\Phi}_x, \widehat{\Phi}_u} || \left[Q^{1/2}, 0; 0, R^{1/2} \right] [\Phi_x + \widehat{\Phi}_x; \Phi_u + \widehat{\Phi}_u] ||^2_F$
  
* Contraints:
   - For **SLS**,  state-feedback constraints is:
     > In theory $[ zI-A, -B_2 ][\Phi_x; \Phi_u] = I$
     
     % > Implementation Phi_x[t + 1] = A Phi_x[t] + B2 Phi_u[t]

   - For **predictive SLS**, state-feedback constriants is:
     > In theory $[ zI-A, -B_2 ][[ \Phi_x, \widehat{\Phi}_x ]; [ \Phi_u, \widehat{\Phi}_u ]] = [ I, 0 ]$
     
     % > Implementation [ Phi_x[t + 1], Phi_hat_x[t - 1] ] = A [ Phi_x[t], Phi_hat_x[t] ] + B2 [ Phi_u[t], Phi_hat_u[t] ]
## Current Progress
For testing, we output K and L_hat and the synthesized counterparts in ``controller_models.py`` and ``algorithms.py`` respectively, and we find intuitive consistency between SLS and Riccati's method in our tracking experiments.
