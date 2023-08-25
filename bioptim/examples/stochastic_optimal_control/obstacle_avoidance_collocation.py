"""
This example aims to replicate the example provided in Gillis 2013: https://doi.org/10.1109/CDC.2013.6761121.
It consists in a mass-point trying to find a time optimal periodic trajectory around super-ellipse obstacles.
The controls are coordinates of a quide-point (the mass is attached to this guide point with a sping).
"""

import matplotlib.pyplot as plt
import casadi as cas
import numpy as np
import pickle

from bioptim import (
    StochasticOptimalControlProgram,
    ObjectiveFcn,
    Solver,
    ObjectiveList,
    OptimalControlProgram,
    NonLinearProgram,
    DynamicsList,
    BoundsList,
    InterpolationType,
    SocpType,
    Node,
    ConstraintList,
    InitialGuessList,
    ControlType,
    PenaltyController,
    PhaseTransitionList,
    PhaseTransitionFcn,
    ConfigureProblem,
    OdeSolver,
    ConstraintFcn,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    StochasticBioModel,
)

from bioptim.examples.stochastic_optimal_control.mass_point_model import MassPointModel

import numpy as np
import matplotlib.pyplot as plt


def superellipse(a=1, b=1, n=2, x_0=0, y_0=0, resolution=100):
    x = np.linspace(-2 * a + x_0, 2 * a + x_0, resolution)
    y = np.linspace(-2 * b + y_0, 2 * b + y_0, resolution)

    X, Y = np.meshgrid(x, y)
    Z = ((X - x_0) / a) ** n + ((Y - y_0) / b) ** n - 1
    return X, Y, Z


def configure_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_new_variable("u", nlp.model.name_u, ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, with_noise=False
        ),
    )


def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_new_variable("u", nlp.model.name_u, ocp, nlp, as_states=False, as_controls=True)

    # Stochastic variables
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=4, n_collocation_points=nlp.model.polynomial_degree+1)
    ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=4)
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, with_noise=False
        ),
    )
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, with_noise=True
        ),
        allow_free_variables=True,
    )


def path_constraint(controller: PenaltyController, super_elipse_index: int, is_robustified: bool = False):
    p_x = controller.states["q"].cx_start[0]
    p_y = controller.states["q"].cx_start[1]

    h = (
        (
            (p_x - controller.model.super_ellipse_center_x[super_elipse_index])
            / controller.model.super_ellipse_a[super_elipse_index]
        )
        ** controller.model.super_ellipse_n[super_elipse_index]
        + (
            (p_y - controller.model.super_ellipse_center_y[super_elipse_index])
            / controller.model.super_ellipse_b[super_elipse_index]
        )
        ** controller.model.super_ellipse_n[super_elipse_index]
        - 1
    )

    out = h

    if is_robustified:
        gamma = 1
        dh_dx = cas.jacobian(h, controller.states.cx_start)
        cov = StochasticBioModel.reshape_to_matrix(controller.stochastic_variables["cov"].cx_start, controller.model.matrix_shape_cov)
        safe_guard = gamma * cas.sqrt(dh_dx @ cov @ dh_dx.T)
        out += safe_guard

    return out


def initialize_circle(polynomial_degree, n_shooting):
    """
    Initialize the positions equally distributed over a circle of radius 3
    """
    q_init = np.zeros((2, (polynomial_degree + 1) * n_shooting + 1))
    for i in range((polynomial_degree + 1) * n_shooting + 1):
        q_init[0, i] = 3 * np.cos(i * 2 * np.pi / (polynomial_degree * n_shooting))
        q_init[1, i] = 3 * np.sin(i * 2 * np.pi / (polynomial_degree * n_shooting))
    return q_init


def prepare_ocp(
    final_time: float,
    n_shooting: int,
    polynomial_degree: int,
) -> OptimalControlProgram:
    """
    Step # 1: Solving the deterministic version of the problem to get the nominal trajectory.
    """

    bio_model = MassPointModel()

    nb_q = bio_model.nb_q
    nb_qdot = bio_model.nb_qdot
    nb_u = bio_model.nb_u

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="u",
        weight=1e-2 / (2 * n_shooting),
        node=Node.ALL_SHOOTING,
        quadratic=True,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(path_constraint, node=Node.ALL, super_elipse_index=0, min_bound=0, max_bound=cas.inf)
    constraints.add(path_constraint, node=Node.ALL, super_elipse_index=1, min_bound=0, max_bound=cas.inf)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_optimal_control_problem,
        dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, with_noise: bio_model.dynamics(
            states,
            controls,
            parameters,
            stochastic_variables,
            nlp,
            with_noise=with_noise,
        ),
        expand=False,
    )

    x_bounds = BoundsList()
    min_q = np.ones((nb_q, 3)) * -cas.inf
    max_q = np.ones((nb_q, 3)) * cas.inf
    min_q[0, 0] = 0  # phi(x) = p_x?
    min_q[0, 2] = 0
    x_bounds.add(
        "q", min_bound=min_q, max_bound=max_q, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
    )
    x_bounds.add(
        "qdot",
        min_bound=[-cas.inf] * nb_qdot,
        max_bound=[cas.inf] * nb_qdot,
        interpolation=InterpolationType.CONSTANT,
    )

    u_bounds = BoundsList()
    u_bounds.add("u", min_bound=[-cas.inf] * nb_u, max_bound=[cas.inf] * nb_u, interpolation=InterpolationType.CONSTANT)

    # Initial guesses
    x_init = InitialGuessList()
    q_init = initialize_circle(polynomial_degree, n_shooting)
    x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.ALL_POINTS)
    x_init.add("qdot", initial_guess=np.zeros((nb_qdot, 1)), interpolation=InterpolationType.CONSTANT)

    u_init = InitialGuessList()
    u_init.add("u", initial_guess=np.zeros((nb_u, 1)), interpolation=InterpolationType.CONSTANT)

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CYCLIC)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transitions,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=polynomial_degree, method="legendre"),
        n_threads=1,
        assume_phase_dynamics=False,
    )


def prepare_socp(
    final_time: float,
    n_shooting: int,
    motor_noise_magnitude: np.ndarray,
    polynomial_degree: int,
    q_init: np.ndarray | None,
    qdot_init: np.ndarray,
    u_init: np.ndarray,
    m_init: np.ndarray | None = None,
    cov_init: np.ndarray | None = None,
    is_robustified: bool = False,
) -> StochasticOptimalControlProgram:
    """
    Step # 2: Solving the stochastic version of the problem to get the stochastic trajectory.
    """

    problem_type = SocpType.COLLOCATION(polynomial_degree=polynomial_degree, method="legendre")

    bio_model = MassPointModel(
        motor_noise_magnitude=motor_noise_magnitude,
        polynomial_degree=polynomial_degree,
    )
    nb_q = bio_model.nb_q
    nb_qdot = bio_model.nb_qdot
    nb_u = bio_model.nb_u

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="u", weight=1e-2 / (2 * n_shooting), node=Node.ALL_SHOOTING, quadratic=True
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(path_constraint, node=Node.ALL, super_elipse_index=0, min_bound=0, max_bound=cas.inf, is_robustified=is_robustified)
    constraints.add(path_constraint, node=Node.ALL, super_elipse_index=1, min_bound=0, max_bound=cas.inf, is_robustified=is_robustified)

    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(MultinodeConstraintFcn.STOCHASTIC_EQUALITY,
                              key="cov",
                              nodes=[n_shooting, 0],
                              nodes_phase=[0, 0])

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_stochastic_optimal_control_problem,
        dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, with_noise: bio_model.dynamics(
            states,
            controls,
            parameters,
            stochastic_variables,
            nlp,
            with_noise=with_noise,
        ),
        expand=False,
    )

    x_bounds = BoundsList()
    min_q = np.ones((nb_q, 3)) * -cas.inf
    max_q = np.ones((nb_q, 3)) * cas.inf
    min_q[0, 0] = 0  # phi(x) = p_x?
    min_q[0, 2] = 0
    x_bounds.add(
        "q", min_bound=min_q, max_bound=max_q, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
    )
    x_bounds.add(
        "qdot",
        min_bound=[-cas.inf] * nb_qdot,
        max_bound=[cas.inf] * nb_qdot,
        interpolation=InterpolationType.CONSTANT,
    )

    u_bounds = BoundsList()
    u_bounds.add("u", min_bound=[-cas.inf] * nb_u, max_bound=[cas.inf] * nb_u, interpolation=InterpolationType.CONSTANT)

    # Initial guesses
    x_init = InitialGuessList()
    x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.ALL_POINTS)
    x_init.add("qdot", initial_guess=qdot_init, interpolation=InterpolationType.ALL_POINTS)

    control_init = InitialGuessList()
    control_init.add("u", initial_guess=u_init, interpolation=InterpolationType.EACH_FRAME)

    s_init = InitialGuessList()
    s_bounds = BoundsList()
    n_m = 4 * 4 * (polynomial_degree + 1)
    n_cov = 4 * 4

    if m_init is None:
        m_init = np.zeros((n_m, n_shooting+1))
    s_init.add(
        "m",
        initial_guess=m_init,
        interpolation=InterpolationType.EACH_FRAME,
    )
    s_bounds.add(
        "m",
        min_bound=[-cas.inf] * n_m,
        max_bound=[cas.inf] * n_m,
        interpolation=InterpolationType.CONSTANT,
    )

    if cov_init is None:
        cov_init = np.zeros((n_cov, n_shooting+1))
    # cov_init = cas.DM_eye(n_states) * np.array([1e-4, 1e-4, 1e-7, 1e-7])
    # idx = 0
    # cov_init_vector = np.zeros((n_states * n_states, 1))
    # for i in range(n_states):
    #     for j in range(n_states):
    #         cov_init_vector[idx] = cov_init[i, j]
    s_init.add(
        "cov",
        initial_guess=cov_init,
        interpolation=InterpolationType.EACH_FRAME,
    )
    s_bounds.add(
        "cov",
        min_bound=[-cas.inf] * n_cov,
        max_bound=[cas.inf] * n_cov,
        interpolation=InterpolationType.CONSTANT,
    )

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CYCLIC)

    return StochasticOptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=control_init,
        s_init=s_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        s_bounds=s_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=1,
        assume_phase_dynamics=False,
        problem_type=problem_type,
        phase_transitions=phase_transitions,
    )


def main():
    """
    Prepare, solve and plot the solution

    The problem is solved in 3 steps with a warm-start between each step:
    step #1: solve the deterministic version
    step #2: solve the stochastic version without the robustified constraint
    step #3: solve the stochastic version with the robustified constraint
    """

    # --- Prepare the ocp --- #
    bio_model = MassPointModel()
    n_shooting = 39
    polynomial_degree = 5
    final_time = 4
    motor_noise_magnitude = np.array([1, 1])

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(1000)

    ocp = prepare_ocp(
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=polynomial_degree,
    )
    sol_ocp = ocp.solve(solver)
    q_deterministic = sol_ocp.states["q"]
    qdot_deterministic = sol_ocp.states["qdot"]
    u_deterministic = sol_ocp.controls["u"]
    data_deterministic = {"q_deterministic": q_deterministic,
                          "qdot_deterministic": qdot_deterministic,
                          "u_deterministic": u_deterministic}
    #save the results
    with open('deterministic.pkl', 'wb') as f:
        pickle.dump(data_deterministic, f)


    socp = prepare_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=polynomial_degree,
        motor_noise_magnitude=motor_noise_magnitude,
        q_init=q_deterministic,
        qdot_init=qdot_deterministic,
        u_init=u_deterministic,
        is_robustified=False,
    )
    sol_socp = socp.solve(solver)
    q_stochastic = sol_socp.states["q"]
    qdot_stochastic = sol_socp.states["qdot"]
    u_stochastic = sol_socp.controls["u"]
    m_stochastic = sol_socp.stochastic_variables["m"]
    cov_stochastic = sol_socp.stochastic_variables["cov"]
    data_stochastic = {"q_stochastic": q_stochastic,
                      "qdot_stochastic": qdot_stochastic,
                      "u_stochastic": u_stochastic,
                      "m_stochastic": m_stochastic,
                      "cov_stochastic": cov_stochastic}
    with open('stochastic.pkl', 'wb') as f:
        pickle.dump(data_stochastic, f)

    rsocp = prepare_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=polynomial_degree,
        motor_noise_magnitude=motor_noise_magnitude,
        q_init=q_stochastic,
        qdot_init=qdot_stochastic,
        u_init=u_stochastic,
        m_init=m_stochastic,
        cov_init=cov_stochastic,
        is_robustified=True,
    )
    sol_rsocp = rsocp.solve(solver)
    q_robustified = sol_rsocp.states["q"]
    qdot_robustified = sol_rsocp.states["qdot"]
    u_robustified = sol_rsocp.controls["u"]
    m_robustified = sol_rsocp.stochastic_variables["m"]
    cov_robustified = sol_rsocp.stochastic_variables["cov"]
    robustified_data = {"q_robustified": q_robustified,
                        "qdot_robustified": qdot_robustified,
                        "u_robustified": u_robustified,
                        "m_robustified": m_robustified,
                        "cov_robustified": cov_robustified}

    with open('robustified.pkl', 'wb') as f:
        pickle.dump(robustified_data, f)


    q_init = initialize_circle(5, n_shooting)
    plt.figure()
    for i in range(2):
        a = bio_model.super_ellipse_a[i]
        b = bio_model.super_ellipse_b[i]
        n = bio_model.super_ellipse_n[i]
        x_0 = bio_model.super_ellipse_center_x[i]
        y_0 = bio_model.super_ellipse_center_y[i]

        X, Y, Z = superellipse(a, b, n, x_0, y_0)

        plt.contourf(X, Y, Z, levels=[-1000, 0], colors=["#DA1984"], alpha=0.5)
        # plt.contour(X, Y, Z, levels=[0], colors='black')

    plt.plot(q_init[0], q_init[1], "-k", label="Initial guess")
    plt.plot(q_deterministic[0], q_deterministic[1], "-g", label="Deterministic")
    plt.plot(q_stochastic[0], q_stochastic[1], "--r", label="Stochastic")
    plt.plot(q_robustified[0], q_robustified[1], "-b", label="Stochastic robustified")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()