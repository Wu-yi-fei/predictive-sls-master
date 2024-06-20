import csv

from model.preSLS import objective
from model.preSLS.algorithms import *
from model.plant_generators import *
from tools.visualization_tools import *
from model.noise_models import *
from model.preSLS.objective import *


predictive = True
space_dimension = 5
FIR_horizon = 24
horizon = 24



# define a customized synthesis algorithm

#  disturbances and predictions
disturbance = []
with open('data/PVscen_1.csv', mode='r') as file:
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
    for i in range(0, len(disturbance)):
        for j in range(5):
            noise._w[i][sys._n_w // 2 - 2 + j] = 1 * disturbance[i-1,j]

    plt.plot(np.arange(24), disturbance[:,:5], marker='o', linestyle='-', markersize=2)
    plt.title("disturbance signals")

    plt.matshow(sys._A, cmap=plt.cm.Reds)
    plt.title("matrix A")
    plt.show()


    if predictive == True:
        simulator = Simulator(
            system=sys,
            noise=noise,
            horizon=sim_horizon,
            prediction=noise
        )
    else:
        simulator = Simulator(
            system=sys,
            noise=noise,
            horizon=sim_horizon,
        )


    # ## (0) basic sls
    # # use SLS controller synthesis algorithm
    # synthesizer1 = My_SynthesisAlgorithm(
    #     system_model=sys
    # )
    # # synthesize controller (the generated controller is actually initialized)
    # # and use the synthesized controller in simulation
    # simulator.setController(
    #     controller=synthesizer1.synthesizeControllerModel()
    # )
    #
    # # run the simulation
    # x_history, y_history, z_history, u_history, w_history = simulator.run()
    #
    # Bu_history = matrix_list_multiplication(sys._B2, u_history)
    # plot_heat_map(x_history, Bu_history, 'Open loop')

    ## (1)centralized sls
    synthesizer = Predictive_SLS(
        system_model=sys,
        FIR_horizon=FIR_horizon,
        predictive=predictive,
    )
    synthesizer << SLS_Obj_LQ()
    # set SLS objective
    simulator.setController(
        controller=synthesizer.synthesizedControllerModel()
    )
    # run the simulation
    x_history, y_history, z_history, u_history, w_history, w_hat1 = simulator.run()
    print(synthesizer.getOptimalObjectiveValue())
    print("OptValue = ", objective.LQC().ObjectiveValue(synthesizer, u_history, x_history))

    Bu_history = matrix_list_multiplication(sys._B2, u_history)
    plot_heat_map(x_history, Bu_history, 'Centralized')
    d_history = noise._w

    plot_time_trajectory(np.array(x_history)[1:49,3:8,0], np.array(Bu_history)[1:49,3:8,0], np.array(d_history)[1:49,3:8])

    # ## (2) d-localized sls
    # synthesizer3 = SLS(
    #     system_model=sys,
    #     FIR_horizon=20
    # )
    # # set SLS objective
    # synthesizer3 << SLS_Obj_H2()
    #
    # dlocalized = SLS_Cons_dLocalized(
    #     act_delay=0,
    #     comm_speed=1,
    #     d=7
    # )
    # synthesizer3 << dlocalized
    #
    # simulator.setController(
    #     controller=synthesizer3.synthesizeControllerModel()
    # )
    #
    # x_history, y_history, z_history, u_history, w_history = simulator.run()
    #
    # Bu_history = matrix_list_multiplication(sys._B2, u_history)
    # plot_heat_map(x_history, Bu_history, 'Localized')


if __name__ == '__main__':
    state_fdbk(horizon)
    keep_showing_figures()
