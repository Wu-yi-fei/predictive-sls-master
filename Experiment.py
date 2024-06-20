import csv
import math
from model.preSLS import objective
from model.preSLS.algorithms import *
from model.plant_generators import *
from tools.visualization_tools import *
from model.noise_models import *
from model.preSLS.objective import *

horizon = 240
FIR_horizon = 240

disturbance = []
with open('data/Windscen_1.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        disturbance.append(lines)
    disturbance = np.array(disturbance, dtype='float32')[1:, 1:]

def plot_trajectory(y,color):

    plt.plot(y[:,0], y[:,1], linewidth=2, color=color, linestyle='dashed', label='Trajectory'+r' $\mathbf{y}$')
    plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.5, prop={'size': 15}, fancybox=True, framealpha=0.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

def plot_track(x, y,context, color):

    plt.plot(x[:,0]+y[:,0], x[:,1]+y[:,1],  linewidth=3, color=color, label=context)
    plt.legend(loc='upper left', scatterpoints=1, frameon=True, labelspacing=0.5, prop={'size': 15}, fancybox=True, framealpha=0.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

def tracking_coordinates(t):

    x = 2 * math.cos(t/38.2) + math.cos(5 * t/38.2)
    y = 2 * math.sin(t/38.2) + math.sin(5 * t/38.2)

    return x, y

def generate_w(mode, A, T):

    w = np.zeros((T, np.shape(A)[0]))

    if mode == 'Tracking':
        for t in range(T):
            y_1, y_2 = tracking_coordinates(t)
            y_3, y_4 = tracking_coordinates(t + 1)

            # Ground-true predictions
            w[t] = np.matmul(A, np.array([y_1, y_2, 0, 0])) - np.array([y_3, y_4, 0, 0])
    return w

def state_sls(mode, sim_horizon):
    A = B = Q = R = None
    sim_horizon = sim_horizon

    if mode == 'Tracking':
        A = np.array([[1, 0, 0, 0.1], [0, 1, 0.1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        B = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        R = np.array([[1.5, 0], [0, 1]])
        # A = np.array([[1, 0, 0.3, 0], [0, 0.4, 0, 0.1], [0, 0, 0, 0], [0, 0, 0, 0]])
        # B = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        # Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # R = np.array([[1, 0], [0, 1]])

    sys = LTI_System(
        n_x=len(A), n_w=len(A), n_u=len(B[0])
    )

    # assign sys._A, sys._B2
    sys._A = A
    sys._B2 = B
    sys._C1 = Q
    sys._D12 = R
    sys._B1 = np.eye(len(A))

    # assign noise
    disturbance = generate_w(mode, A, horizon)
    noise = FixedNoiseVector(n_w=sys._n_w, horizon=sim_horizon, FIR_horizon=FIR_horizon)
    noise.generateNoiseFromNoiseModel(cls=ZeroNoise)
    for i in range(0, len(disturbance)):
        for j in range(4):
            noise._w[i][j] = disturbance[i, j]
    y = np.zeros((horizon, 2))
    for t in range(horizon):
        y_1, y_2 = tracking_coordinates(t)
        y[t] = [y_1, y_2]
    plot_trajectory(y, 'black')
    plt.grid()
    plt.show()

    ## (1) Causal Riccati
    print()
    print("--------------Causal Riccati--------------")
    simulator_1 = Causal_Riccati(
            system=sys,
            noise=noise,
            horizon=sim_horizon,
    )
    simulator_1.setController(
            controller=Causal_StateFeedback_Controller(sys._A, sys._B2, sys._C1, sys._D12, horizon)
        )

    x_history_1, u_history_1, w_history_1 = simulator_1.run(sys._A, sys._B2, sys._C1, sys._D12)

    ## (2) Optimal Riccati
    print()
    print("--------------Optimal Riccati--------------")
    simulator_2 = Noncausal_Riccati_Optimal(
        system=sys,
        noise=noise,
        horizon=sim_horizon,
        prediction=noise
    )
    simulator_2.setController(
        controller=Noncausal_StateFeedback_Controller(sys._A, sys._B2, sys._C1, sys._D12, horizon)
    )

    x_history_2, u_history_2, w_history_2, w_hat_2 = simulator_2.run(sys._A, sys._B2, sys._C1, sys._D12)

    ## (3) Centralized SLS
    print()
    print("--------------SLS--------------")
    simulator_3 = Simulator(
        system=sys,
        noise=noise,
        horizon=sim_horizon,
    )
    synthesizer = Predictive_SLS(
        system_model=sys,
        FIR_horizon=FIR_horizon,
        predictive=False,
    )

    synthesizer << SLS_Obj_LQ()
    # set SLS objective
    simulator_3.setController(
        controller=synthesizer.synthesizedControllerModel()
    )
    # run the simulation
    x_history_3, y_history_3, z_history_3, u_history_3, w_history_3, w_hat3 = simulator_3.run()
    Bu_history_3 = matrix_list_multiplication(sys._B2, u_history_3)

    ## (4) Centralized Predictive SLS
    print()
    print("--------------Predictive SLS--------------")
    simulator_4 = Simulator(
            system=sys,
            noise=noise,
            horizon=sim_horizon,
            prediction=noise
        )
    synthesizer1 = Predictive_SLS(
        system_model=sys,
        FIR_horizon=FIR_horizon,
        predictive=True,
    )

    synthesizer1 << SLS_Obj_LQ()
    # set SLS objective
    simulator_4.setController(
        controller=synthesizer1.synthesizedControllerModel()
    )
    # run the simulation
    x_history_4, y_history_4, z_history_4, u_history_4, w_history_4, w_hat4 = simulator_4.run()
    Bu_history_4 = matrix_list_multiplication(sys._B2, u_history_4)

    plot_heat_map(x_history_3, Bu_history_3, 'SLS')
    plot_heat_map(x_history_4, Bu_history_4, 'Predictive SLS')

    print()
    print("OptValueRiccati = ", objective.LQC(sys._C1, sys._D12).ObjectiveValue(us=u_history_1, xs=x_history_1))
    print("OptValueOptimal = ", objective.LQC(sys._C1, sys._D12).ObjectiveValue(us=u_history_2, xs=x_history_2))
    print("OptValueSLS = ", objective.LQC(sys._C1, sys._D12).ObjectiveValue(us=u_history_3, xs=x_history_3))
    print("OptValuePreSLS = ", objective.LQC(sys._C1, sys._D12).ObjectiveValue(us=u_history_4, xs=x_history_4))
    plot_track(np.array(x_history_1)[:240, :, 0],y[:240], r'Riccati $t\in [0,240]$','blue')
    plot_track(np.array(x_history_2)[:240, :, 0], y[:240], r'Predictive Riccati $t\in [0,240]$', 'gray')
    plot_trajectory(y, 'black')

    plt.show()

    plot_track(np.array(x_history_3)[:240, :, 0], y[:240], r'SLS $t\in [0,240]$', 'green')
    plot_track(np.array(x_history_4)[:240, :, 0], y[:240], r'Predictive SLS $t\in [0,240]$', 'red')
    plot_trajectory(y, 'black')
    d_history = noise._w

    plot_time_trajectory(np.array(x_history_4)[:240, :, 0], np.array(Bu_history_4)[:240, :, 0],
                         np.array(d_history)[:240, :])

    plt.show()
if __name__ == '__main__':
    state_sls('Tracking', horizon)
    keep_showing_figures()