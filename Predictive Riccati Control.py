import csv

from model.preSLS import objective
from model.controller_models import *
from model.plant_generators import *
from tools.visualization_tools import *
from model.noise_models import *
from model.preSLS.objective import *


predictive = True

space_dimension = 5
FIR_horizon = 48
horizon = 48


#  disturbances and predictions
disturbance = []
with open('data/Windscen_9.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        disturbance.append(lines)
    disturbance = np.array(disturbance, dtype='float32')[1:, 1:]


# state feedback control process
def state_fdbk(sim_horizon):
    sim_horizon = sim_horizon
    sys = LTI_System(
        n_x=space_dimension, n_w=space_dimension
    )

    # generate sys._A, sys._B2
    generate_doubly_stochastic_chain(
        system_model=sys,
        rho=1,
        actuator_density=1,
        alpha=0.2
    )
    generate_BCD_and_zero_initialization(sys)

    # generate noise
    noise = FixedNoiseVector(n_w=sys._n_w, horizon=sim_horizon, FIR_horizon=FIR_horizon)
    noise.generateNoiseFromNoiseModel(cls=ZeroNoise)
    for i in range(len(disturbance)):
        for j in range(5):
            noise._w[i][sys._n_w // 2 - 2 + j] = 1 * disturbance[i,j]

    plt.plot(np.arange(24), disturbance[:,:5], marker='o', linestyle='-', markersize=2)
    plt.title("disturbance signals")

    plt.matshow(sys._A, cmap=plt.cm.Reds)
    plt.title("matrix A")
    plt.show()

    if predictive == True:
        simulator = Noncausal_Riccati_Optimal(
            system=sys,
            noise=noise,
            horizon=sim_horizon,
            prediction=noise
        )

        simulator.setController(
            controller=Noncausal_StateFeedback_Controller(sys._A, sys._B2, sys._C1, sys._D12, horizon)
        )
        x_history, u_history, w_history, w_hat1 = simulator.run(sys._A, sys._B2, sys._C1, sys._D12)
        print("OptValue = ", objective.LQC(sys._C1, sys._D12).ObjectiveValue(us=u_history, xs=x_history))

        Bu_history = matrix_list_multiplication(sys._B2, u_history)
        plot_heat_map(x_history, Bu_history, 'Centralized')
        d_history = np.concatenate(
            (disturbance[:, :sys._n_w], np.zeros(shape=[horizon - disturbance.shape[0], sys._n_w])), axis=0)

        plot_time_trajectory(np.array(x_history)[:48, 3:8, 0], np.array(Bu_history)[:48, 3:8, 0],
                             np.array(d_history)[:48, 3:8])

    else:
        simulator = Causal_Riccati(
            system=sys,
            noise=noise,
            horizon=sim_horizon,
        )

        simulator.setController(
            controller=Causal_StateFeedback_Controller(sys._A, sys._B2, sys._C1, sys._D12, horizon)
        )
        x_history, u_history, w_history = simulator.run(sys._A, sys._B2, sys._C1, sys._D12)
        print(objective.LQC(sys._C1, sys._D12).ObjectiveValue(us=u_history, xs=x_history))

        Bu_history = matrix_list_multiplication(sys._B2, u_history)
        plot_heat_map(x_history, Bu_history, 'Centralized')
        d_history = np.concatenate(
            (disturbance[:, :sys._n_w], np.zeros(shape=[horizon - disturbance.shape[0], sys._n_w])), axis=0)

        plot_time_trajectory(np.array(x_history)[:49, 3:8, 0], np.array(Bu_history)[:49, 3:8, 0],
                             np.array(d_history)[:49, 3:8])

if __name__ == '__main__':
    state_fdbk(horizon)
    keep_showing_figures()
