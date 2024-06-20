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

## Mathematical interpretation of the implementation
* Objective Fucntion:
  - For SLS, a convex optimization problem we have implemented is as follows:
  > SLS = min_{Phi_x, Phi_u} || [[Q^{1/2}, 0]; [0, R^{1/2}]] [[Phi_x]; [Phi_u]] ||^2_Frob
  - For predictive SLS, a convex optimization problem we have implemented is as follows:
  > PSLS = min_{Phi_x, Phi_u, Phi_hat_x, Phi_hat_u} || [[Q^{1/2}, 0]; [0, R^{1/2}]] [[Phi_x + Phi_hat_x]; [Phi_u + Phi_hat_u]] ||^2_Frob
## Current progress
For testing, we output K and L_hat and the synthesized counterparts in ``controller_models.py`` and ``algorithms.py`` respectively, unfortunately not exactly as expected.
