"""
An example of how to use multi-start to find local minima from different initial guesses.
This example is a variation of the pendulum example in getting_started/pendulum.py.
"""
import pickle
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    CostType,
    Solver,
    InterpolationType,
    MultiStart,
    Solution,
    MagnitudeType,
)


def prepare_ocp(
    bio_model_path: str,
    final_time: float,
    n_shooting: int,
    seed: int = 0,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    bio_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    seed: int
        The seed to use for the random initial guess

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(bio_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuess([0] * (n_q + n_qdot), interpolation=InterpolationType.CONSTANT)
    x_init = x_init.add_noise(
        bounds=x_bounds,
        magnitude=0.5,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=n_shooting + 1,
        seed=seed,
    )

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    u_init = InitialGuess([0] * n_tau, interpolation=InterpolationType.CONSTANT)
    u_init = u_init.add_noise(
        bounds=u_bounds,
        magnitude=0.5,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=n_shooting,
        seed=seed,
    )

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        n_threads=1,  # You cannot use multi-threading for the resolution of the ocp with multi-start
    )

    ocp.add_plot_penalty(CostType.ALL)

    return ocp


def save_results(sol: Solution, biorbd_model_path: str, final_time: float, n_shooting: int, seed: int):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    seed: int
        The seed to use for the random initial guess
    """
    # OptimalControlProgram.save(sol, f"solutions/pendulum_multi_start_random{seed}.bo", stand_alone=True)
    states = sol.states["all"]
    with open(f"pendulum_multi_start_random_states_{n_shooting}_{seed}.pkl", "wb") as file:
        pickle.dump(states, file)


def prepare_multi_start(bio_model_path: list, final_time: list, n_shooting: list, seed: list) -> MultiStart:
    """
    The initialization of the multi-start
    """
    return MultiStart(
        prepare_ocp,
        solver=Solver.IPOPT(show_online_optim=False),  # You cannot use show_online_optim with multi-start
        post_optimization_callback=save_results,
        n_pools=4,
        bio_model_path=bio_model_path,
        final_time=final_time,
        n_shooting=n_shooting,
        seed=seed,
    )


def main():
    # --- Prepare the multi-start and run it --- #
    multi_start = prepare_multi_start(
        bio_model_path=["models/pendulum.bioMod"], final_time=[1], n_shooting=[30, 40, 50], seed=[0, 1, 2, 3]
    )
    multi_start.solve()


if __name__ == "__main__":
    main()
