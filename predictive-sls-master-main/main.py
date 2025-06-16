import csv

import numpy as np
from numpy.ma.core import array

from model.PredSLS import objective
from model.PredSLS.algorithms import *
from model.plant_generators import *
from tools.visualization_tools import *
from model.noise_models import *
from model.PredSLS.objective import *
import time

# state feedback control process
def distributed_control(sim_horizon, agent_number, predictive, d_locality, comm_speed, prediction_error=0, order=0):
    Phi_x = []
    Phi_u = []
    Phi_hat_x = []
    Phi_hat_u = []
    sys = LTI_System(n_x=agent_number, n_w=agent_number)
    generate_doubly_stochastic_chain(
        system_model=sys,
        rho=1,
        actuator_density=1,
        alpha=0.2
    )
    generate_BCD_and_zero_initialization(sys)
    np.random.seed(order)

    A = np.array([[1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1]])

    sys._B2 = np.array([[1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1]])

    sys._A = A

    noise = FixedNoiseVector(n_w=sys._n_w, horizon=horizon, FIR_horizon=horizon)
    noise.generateNoiseFromNoiseModel(cls=ZeroNoise)
    for i in range(horizon-8):
        for j in range(len(A)):
            noise._w[i+1][j] = np.random.normal(0, 0.5)

    noise_prediction = None
    if predictive:
        noise_prediction = FixedNoiseVector(n_w=sys._n_w, horizon=horizon, FIR_horizon=horizon)
        noise_prediction.generateNoiseFromNoiseModel(cls=ZeroNoise)
        for i in range(horizon):
            for j in range(len(A)):
                if j == 0:
                    noise_prediction._w[i][j] = noise._w[i][j] + prediction_error
                else:
                    noise_prediction._w[i][j] = noise._w[i][j]

    # w = np.array(noise._w)
    # w_hat = np.array(noise_prediction._w)
    objs = 0
    for agent_i in range(agent_number):

        synthesizer = Distributed_SLS_Predictive(
            system_model=sys,
            FIR_horizon=FIR_horizon,
            predictive=predictive,
            agent=agent_i,
            d_locality = d_locality
        )
        synthesizer << SLS_Obj_LQ_Dist()
        if predictive:
            phi_x, phi_hat_x, phi_u, phi_hat_u, obj = synthesizer.synthesizedControllerModel()
            Phi_x.append(phi_x)
            Phi_u.append(phi_u)
            Phi_hat_x.append(phi_hat_x)
            Phi_hat_u.append(phi_hat_u)
        else:
            phi_x, phi_u, obj = synthesizer.synthesizedControllerModel()
            Phi_x.append(phi_x)
            Phi_u.append(phi_u)

        objs += obj
    Phi_x = np.array(Phi_x).transpose(1,2,0,3).reshape((horizon + 1) * (horizon + 1), space_dimension, -1)
    Phi_u = np.array(Phi_u).transpose(1,2,0,3).reshape((horizon + 1) * (horizon + 1), space_dimension, -1)
    if predictive:
        Phi_hat_x = np.array(Phi_hat_x).transpose(1, 2, 0, 3).reshape((horizon) * (horizon), space_dimension, -1)
        Phi_hat_u = np.array(Phi_hat_u).transpose(1, 2, 0, 3).reshape((horizon) * (horizon), space_dimension, -1)

    if predictive:
        controller = Predictive_SLS_StateFeedback_Controller(
            n_x=sys._n_x,
            n_u=sys._n_u,
            FIR_horizon=horizon
        )
        controller._Phi_x = [None] * (horizon + 1) * (horizon + 1)
        controller._Phi_u = [None] * (horizon + 1) * (horizon + 1)
        controller._Phi_hat_x = [None] * horizon * horizon
        controller._Phi_hat_u = [None] * horizon * horizon
        for t in range((horizon + 1) * (horizon + 1)):
            controller._Phi_x[t] = Phi_x[t]
            controller._Phi_u[t] = Phi_u[t]
        for t in range(horizon * horizon):
            controller._Phi_hat_x[t] = Phi_hat_x[t]
            controller._Phi_hat_u[t] = Phi_hat_u[t]
    else:
        controller = SLS_StateFeedback_Controller(
            n_x=sys._n_x,
            n_u=sys._n_u,
            FIR_horizon=horizon
        )
        controller._Phi_x = [None] * (horizon + 1) * (horizon + 1)
        controller._Phi_u = [None] * (horizon + 1) * (horizon + 1)
        for t in range((horizon + 1) * (horizon + 1)):
            controller._Phi_x[t] = Phi_x[t]
            controller._Phi_u[t] = Phi_u[t]

    simulator = Simulator(
        system=sys,
        noise=noise,
        horizon=horizon,
        prediction=noise_prediction  # noise_prediction
    )
    simulator.setController(
        controller=controller
    )

    # run the simulation
    x_history, y_history, z_history, u_history, w_history = simulator.run()

    print("LQR Objective=", objective.LQC(sys._A, sys._B2, sys._C1, sys._D12).ObjectiveValue(us=u_history, xs=x_history[:]))

    # plot
    Bu_history = matrix_list_multiplication(sys._B2, u_history)
    plot_heat_map(x_history[:], Bu_history[:], 'Trajectories')
    plot_time_trajectory(np.array(x_history)[:, :, 0], np.array(Bu_history)[:, :, 0],
                         np.array(noise.getNoiseTrajectroy1())[:, :])
    return objective.LQC(sys._A, sys._B2, sys._C1, sys._D12).ObjectiveValue(us=u_history, xs=x_history[:])




if __name__ == '__main__':
    predictive = True
    space_dimension = 16
    FIR_horizon = 20
    horizon = 20
    comm_speed = 100

    iter = 0

    for order in range(100):
        prediction_error = 0
        while prediction_error <= 1.0:
            if prediction_error == 0:
                print("--------------------", iter, "| centralized | error=%s-------------------"%prediction_error)
                cost_c = distributed_control(horizon, space_dimension, predictive, 222, comm_speed,
                                             prediction_error=prediction_error, order=order)


            for d_locality in range(1,space_dimension):
                print("--------------------", iter, "| d=%s | error=%s-------------------" % (d_locality, prediction_error))
                cost_d = distributed_control(horizon, space_dimension, predictive, d_locality, comm_speed,
                                             prediction_error=prediction_error, order=order)
                iter+=1
            if prediction_error == 0:
                prediction_error = 0.1
            else:
                prediction_error = prediction_error + 0.1