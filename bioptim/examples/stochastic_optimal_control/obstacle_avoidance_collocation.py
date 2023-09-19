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
from bioptim.examples.stochastic_optimal_control.common import (
    get_m_init,
    get_cov_init_collocations,
    get_cov_init_dms,
    get_cov_init_slicot,
    test_matrix_semi_definite_positiveness,
    test_robustified_constraint_value,
    test_eigen_values,
)


def superellipse(a=1, b=1, n=2, x_0=0, y_0=0, resolution=100):
    x = np.linspace(-2 * a + x_0, 2 * a + x_0, resolution)
    y = np.linspace(-2 * b + y_0, 2 * b + y_0, resolution)

    X, Y = np.meshgrid(x, y)
    Z = ((X - x_0) / a) ** n + ((Y - y_0) / b) ** n - 1
    return X, Y, Z

def draw_cov_ellipse(cov, pos, ax):
    """
    Draw an ellipse representing the covariance at a given point.
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(vals)
    ellip = plt.matplotlib.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, color="b", alpha=0.1)

    ax.add_patch(ellip)
    return ellip


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


def configure_stochastic_optimal_control_problem_collocations(ocp: OptimalControlProgram, nlp: NonLinearProgram):
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


def configure_stochastic_optimal_control_problem_dms(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_new_variable("u", nlp.model.name_u, ocp, nlp, as_states=False, as_controls=True)

    # Stochastic variables
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


def initialize_circle(n_points):
    """
    Initialize the positions equally distributed over a circle of radius 3
    """
    q_init = np.zeros((2, n_points))
    for i in range(n_points):
        q_init[0, i] = 3 * np.sin(i * 2 * np.pi / (n_points - 1))
        q_init[1, i] = 3 * np.cos(i * 2 * np.pi / (n_points - 1))
    return q_init


def prepare_ocp(
    final_time: float,
    n_shooting: int,
    polynomial_degree: int,
    socp_type: SocpType = SocpType.COLLOCATION(polynomial_degree=5, method="legendre"),
) -> OptimalControlProgram:
    """
    Step # 1: Solving the deterministic version of the problem to get the nominal trajectory.
    """

    bio_model = MassPointModel(socp_type=socp_type)

    nb_q = bio_model.nb_q
    nb_qdot = bio_model.nb_qdot
    nb_u = bio_model.nb_u

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTROL,
        key="u",
        weight=1e-2 / (2 * n_shooting),
        node=Node.ALL_SHOOTING,
        quadratic=True,
    )

    # Constraints
    constraints = ConstraintList()
    epsilon = 0
    constraints.add(path_constraint, node=Node.ALL, super_elipse_index=0, min_bound=0+epsilon, max_bound=cas.inf, quadratic=False)
    constraints.add(path_constraint, node=Node.ALL, super_elipse_index=1, min_bound=0+epsilon, max_bound=cas.inf, quadratic=False)

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
    max_q[0, 0] = 0
    x_bounds.add(
        "q", min_bound=min_q, max_bound=max_q, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
    )
    x_bounds.add(
        "qdot",
        min_bound=[-30] * nb_qdot,
        max_bound=[30] * nb_qdot,
        interpolation=InterpolationType.CONSTANT,
    )

    u_bounds = BoundsList()
    u_bounds.add("u", min_bound=[-10] * nb_u, max_bound=[10] * nb_u, interpolation=InterpolationType.CONSTANT)

    # Initial guesses
    x_init = InitialGuessList()
    if isinstance(socp_type, SocpType.COLLOCATION):
        n_points = n_shooting * (polynomial_degree + 1) + 1
        q_init = initialize_circle(n_points)
        x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.ALL_POINTS)
    else:
        n_points = n_shooting + 1
        q_init = initialize_circle(n_points)
        x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=np.zeros((nb_qdot, 1)), interpolation=InterpolationType.CONSTANT)

    u_init = InitialGuessList()
    u_init.add("u", initial_guess=np.zeros((nb_u, 1)), interpolation=InterpolationType.CONSTANT)

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CYCLIC)

    if isinstance(socp_type, SocpType.COLLOCATION):
        ode_solver = OdeSolver.COLLOCATION(polynomial_degree=socp_type.polynomial_degree, method=socp_type.method)
    elif isinstance(socp_type, SocpType.DMS):
        ode_solver = OdeSolver.RK4()
    else:
        raise RuntimeError("Invalid socp_type")

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
        ode_solver=ode_solver,
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
    socp_type: SocpType = SocpType.COLLOCATION(polynomial_degree=5, method="legendre"),
) -> StochasticOptimalControlProgram:
    """
    Step # 2: Solving the stochastic version of the problem to get the stochastic trajectory (is_robustified = False).
    Step # 3: Solving the stochastic version of the problem to get the stochastic trajectory (is_robustified = True).
    """

    problem_type = socp_type

    bio_model = MassPointModel(
        socp_type=problem_type,
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
        ObjectiveFcn.Mayer.MINIMIZE_CONTROL, key="u", weight=1e-2 / (2 * n_shooting), node=Node.ALL_SHOOTING, quadratic=True
    )

    # Constraints
    constraints = ConstraintList()
    epsilon = 0 if not is_robustified else 0
    constraints.add(path_constraint, node=Node.ALL, super_elipse_index=0, min_bound=0+epsilon, max_bound=cas.inf, is_robustified=is_robustified, quadratic=False)
    constraints.add(path_constraint, node=Node.ALL, super_elipse_index=1, min_bound=0+epsilon, max_bound=cas.inf, is_robustified=is_robustified, quadratic=False)
    # constraints.add(ConstraintFcn.SYMMETRIC_MATRIX, node=Node.START, key="cov")
    # constraints.add(ConstraintFcn.SEMIDEFINITE_POSITIVE_MATRIX, node=Node.START, key="cov", min_bound=0, max_bound=cas.inf, quadratic=False)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_stochastic_optimal_control_problem_collocations if isinstance(socp_type, SocpType.COLLOCATION) else configure_stochastic_optimal_control_problem_dms,
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
    min_q[0, 0] = 0
    max_q[0, 0] = 0
    x_bounds.add(
        "q", min_bound=min_q, max_bound=max_q, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
    )
    x_bounds.add(
        "qdot",
        min_bound=[-30] * nb_qdot,
        max_bound=[30] * nb_qdot,
        interpolation=InterpolationType.CONSTANT,
    )

    u_bounds = BoundsList()
    u_bounds.add("u", min_bound=[-10] * nb_u, max_bound=[10] * nb_u, interpolation=InterpolationType.CONSTANT)

    # Initial guesses
    x_init = InitialGuessList()
    if isinstance(socp_type, SocpType.COLLOCATION):
        x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot", initial_guess=qdot_init, interpolation=InterpolationType.ALL_POINTS)
    else:
        x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot", initial_guess=qdot_init, interpolation=InterpolationType.EACH_FRAME)

    control_init = InitialGuessList()
    control_init.add("u", initial_guess=u_init, interpolation=InterpolationType.EACH_FRAME)

    s_init = InitialGuessList()
    s_bounds = BoundsList()
    n_cov = 4 * 4
    n_stochastic = n_cov

    if isinstance(socp_type, SocpType.COLLOCATION):
        n_m = 4 * 4 * (polynomial_degree + 1)
        n_stochastic += n_m
        if m_init is None:
            m_init = get_m_init(bio_model, n_stochastic, n_shooting, final_time, polynomial_degree, q_init, qdot_init, u_init)
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
        cov_init_matrix = cas.DM_eye(nb_q + nb_qdot) * 0.01
        shape_0, shape_1 = cov_init_matrix.shape[0], cov_init_matrix.shape[1]
        cov_0 = np.zeros((shape_0 * shape_1, 1))
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                cov_0[shape_0 * s1 + s0] = cov_init_matrix[s0, s1]
        if isinstance(socp_type, SocpType.COLLOCATION):
            cov_init = get_cov_init_collocations(bio_model,
                                    n_shooting,
                                    n_stochastic,
                                    polynomial_degree,
                                    final_time,
                                    q_init,
                                    qdot_init,
                                    u_init,
                                    m_init,
                                    cov_0,
                                    motor_noise_magnitude)
        else:
            # cov_init = get_cov_init_dms(bio_model,
            #                         n_shooting,
            #                         n_stochastic,
            #                         final_time,
            #                         q_init,
            #                         qdot_init,
            #                         u_init,
            #                         cov_0,
            #                         motor_noise_magnitude)
            cov_init = get_cov_init_slicot(bio_model,
                                    n_shooting,
                                    n_stochastic,
                                    final_time,
                                    q_init,
                                    qdot_init,
                                    u_init,
                                    cov_0,
                                    motor_noise_magnitude)
    s_init.add(
        "cov",
        initial_guess=cov_init,
        interpolation=InterpolationType.EACH_FRAME,
    )
    # s_bounds.add(
    #     "cov",
    #     min_bound=[-cas.inf] * n_cov,
    #     max_bound=[cas.inf] * n_cov,
    #     interpolation=InterpolationType.CONSTANT,
    # )
    cov_min = np.ones((n_cov, 3)) * -cas.inf
    cov_max = np.ones((n_cov, 3)) * cas.inf
    cov_min[:, 0] = cov_init[:, 0]
    cov_max[:, 0] = cov_init[:, 0]
    s_bounds.add(
        "cov",
        min_bound=cov_min,
        max_bound=cov_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    for i in range(n_shooting + 1):
        if not test_matrix_semi_definite_positiveness(cov_init[:, i]):
            print(
                f"Initial guess for cov is not semi-definite positive, something went wrong at the {i}th node.")

        if not test_eigen_values(cov_init[:, i]):
            print(f"Something went wrong at the {i}th node. (Eigen values)")

        if not test_robustified_constraint_value(bio_model, q_init, qdot_init, cov_init):
            print(
                f"Initial guess for cov is incompatible with the robustified constraint, something went wrong at the {i}th node.")

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CYCLIC)
    # phase_transitions.add(PhaseTransitionFcn.COVARIANCE_CONTINUOUS, phase_pre_idx=0)

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
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=4,
        assume_phase_dynamics=True,
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
    run_step_1 = False
    run_step_2 = True
    run_step_3 = False  # True

    # --- Prepare the ocp --- #
    # socp_type = SocpType.COLLOCATION(polynomial_degree=5, method="legendre")
    socp_type = SocpType.DMS()
    bio_model = MassPointModel(socp_type=socp_type)
    n_shooting = 39
    polynomial_degree = 5
    final_time = 4
    motor_noise_magnitude = np.array([1, 1])

    # Solver parameters
    # TODO: include SLICOT solver (for solving ill-conditioned, ill-scaled, large-scale problems by preserving the stucture of the problem)
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(0) # 1000
    # solver._nlp_scaling_method = "None"

    if run_step_1:
        ocp = prepare_ocp(
            final_time=final_time,
            n_shooting=n_shooting,
            polynomial_degree=polynomial_degree,
            socp_type=socp_type,
        )
        sol_ocp = ocp.solve(solver)
        q_deterministic = sol_ocp.states["q"]
        qdot_deterministic = sol_ocp.states["qdot"]
        u_deterministic = sol_ocp.controls["u"]
        time_deterministic = sol_ocp.parameters["time"][0][0]
        data_deterministic = {"q_deterministic": q_deterministic,
                              "qdot_deterministic": qdot_deterministic,
                              "u_deterministic": u_deterministic,
                              "time_deterministic": time_deterministic}
        #save the results
        with open('deterministic.pkl', 'wb') as f:
            pickle.dump(data_deterministic, f)
        # sol_ocp.graphs()
    else:
        with open('deterministic.pkl', 'rb') as f:
            data_deterministic = pickle.load(f)
            q_deterministic = data_deterministic["q_deterministic"]
            qdot_deterministic = data_deterministic["qdot_deterministic"]
            u_deterministic = data_deterministic["u_deterministic"]
            time_deterministic = data_deterministic["time_deterministic"]


    if run_step_2:
        socp = prepare_socp(
            final_time=time_deterministic,
            n_shooting=n_shooting,
            polynomial_degree=polynomial_degree,
            motor_noise_magnitude=motor_noise_magnitude,
            q_init=q_deterministic,
            qdot_init=qdot_deterministic,
            u_init=u_deterministic,
            is_robustified=False,
            socp_type=socp_type,
        )
        sol_socp = socp.solve(solver)
        q_stochastic = sol_socp.states["q"]
        qdot_stochastic = sol_socp.states["qdot"]
        u_stochastic = sol_socp.controls["u"]
        time_stochastic = sol_socp.parameters["time"][0][0]
        if isinstance(socp_type, SocpType.COLLOCATION):
            m_stochastic = sol_socp.stochastic_variables["m"]
        else:
            m_stochastic = None
        cov_stochastic = sol_socp.stochastic_variables["cov"]
        data_stochastic = {"q_stochastic": q_stochastic,
                          "qdot_stochastic": qdot_stochastic,
                          "u_stochastic": u_stochastic,
                          "m_stochastic": m_stochastic,
                          "cov_stochastic": cov_stochastic,
                          "time_stochastic": time_stochastic}
        with open('stochastic.pkl', 'wb') as f:
            pickle.dump(data_stochastic, f)
        # sol_socp.graphs()
    else:
        with open('stochastic.pkl', 'rb') as f:
            data_stochastic = pickle.load(f)
            q_stochastic = data_stochastic["q_stochastic"]
            qdot_stochastic = data_stochastic["qdot_stochastic"]
            u_stochastic = data_stochastic["u_stochastic"]
            m_stochastic = data_stochastic["m_stochastic"]
            cov_stochastic = data_stochastic["cov_stochastic"]
            time_stochastic = data_stochastic["time_stochastic"]

    for i in range(n_shooting + 1):
        if not test_matrix_semi_definite_positiveness(cov_stochastic[:, i]):
            print(f"Something went wrong at the {i}th node. (Semi-definiteness)")

        if not test_eigen_values(cov_stochastic[:, i]):
            print(f"Something went wrong at the {i}th node. (Eigen values)")

        # if not test_robustified_constraint_value(bio_model, q_stochastic, qdot_stochastic, cov_stochastic):
        #     print(f"Something went wrong at the {i}th node. (Constraint evaluation)")

    if run_step_3:
        # rsocp = prepare_socp(
        #     final_time=final_time,
        #     n_shooting=n_shooting,
        #     polynomial_degree=polynomial_degree,
        #     motor_noise_magnitude=motor_noise_magnitude,
        #     q_init=q_stochastic + 1e-10,
        #     qdot_init=qdot_stochastic + 1e-10,
        #     u_init=u_stochastic + 1e-10,
        #     m_init=m_stochastic + 1e-15,
        #     cov_init=cov_stochastic + 1e-10,
        #     is_robustified=True,
        #     socp_type=socp_type,
        # )

        rsocp = prepare_socp(
            final_time=time_stochastic,
            n_shooting=n_shooting,
            polynomial_degree=polynomial_degree,
            motor_noise_magnitude=motor_noise_magnitude,
            q_init=q_stochastic,
            qdot_init=qdot_stochastic,
            u_init=u_stochastic,
            m_init=m_stochastic,
            cov_init=cov_stochastic,
            is_robustified=True,
            socp_type=socp_type,
        )

        sol_rsocp = rsocp.solve(solver)
        q_robustified = sol_rsocp.states["q"]
        qdot_robustified = sol_rsocp.states["qdot"]
        u_robustified = sol_rsocp.controls["u"]
        # time_robustified = sol_rsocp.parameters["time"][0][0]
        if isinstance(socp_type, SocpType.COLLOCATION):
            m_robustified = sol_rsocp.stochastic_variables["m"]
        else:
            m_robustified = None
        cov_robustified = sol_rsocp.stochastic_variables["cov"]
        robustified_data = {"q_robustified": q_robustified,
                            "qdot_robustified": qdot_robustified,
                            "u_robustified": u_robustified,
                            # "time_robustified": time_robustified,
                            "m_robustified": m_robustified,
                            "cov_robustified": cov_robustified}

        with open('robustified.pkl', 'wb') as f:
            pickle.dump(robustified_data, f)
        # sol_rsocp.graphs()
    else:
        with open('robustified.pkl', 'rb') as f:
            robustified_data = pickle.load(f)
            q_robustified = robustified_data["q_robustified"]
            qdot_robustified = robustified_data["qdot_robustified"]
            u_robustified = robustified_data["u_robustified"]
            # time_robusified = robustified_data["time_robustified"]
            m_robustified = robustified_data["m_robustified"]
            cov_robustified = robustified_data["cov_robustified"]

    if isinstance(socp_type, SocpType.COLLOCATION):
        q_init = initialize_circle(6*n_shooting+1)
    else:
        q_init = initialize_circle(n_shooting + 1)
    fig, ax = plt.subplots(1, 1)
    for i in range(2):
        a = bio_model.super_ellipse_a[i]
        b = bio_model.super_ellipse_b[i]
        n = bio_model.super_ellipse_n[i]
        x_0 = bio_model.super_ellipse_center_x[i]
        y_0 = bio_model.super_ellipse_center_y[i]

        X, Y, Z = superellipse(a, b, n, x_0, y_0)

        ax.contourf(X, Y, Z, levels=[-1000, 0], colors=["#DA1984"], alpha=0.5)
        # ax.contour(X, Y, Z, levels=[0], colors='black')

    ax.plot(q_init[0], q_init[1], "-k", label="Initial guess")
    ax.plot(q_deterministic[0], q_deterministic[1], "-g", label="Deterministic")
    ax.plot(q_stochastic[0], q_stochastic[1], "--r", label="Stochastic")
    # ax.plot(q_robustified[0], q_robustified[1], "-b", label="Stochastic robustified")
    if isinstance(socp_type, SocpType.COLLOCATION):
        nb_points = polynomial_degree + 1
    else:
        nb_points = 1
    for j in range(cov_stochastic.shape[1]):
        ax.plot(q_stochastic[0, j*nb_points], q_stochastic[1, j*nb_points], "or", markersize=2)
        cov_reshaped_to_matrix = np.zeros(bio_model.matrix_shape_cov)
        for s_0 in range(bio_model.matrix_shape_cov[0]):
            for s_1 in range(bio_model.matrix_shape_cov[1]):
                cov_reshaped_to_matrix[s_0, s_1] = cov_stochastic[bio_model.matrix_shape_cov[0] * s_1 + s_0, j]
        draw_cov_ellipse(cov_reshaped_to_matrix[:2, :2], q_stochastic[:, j*nb_points], ax)
        tempo_cov_reshaped_to_matrix = np.zeros(bio_model.matrix_shape_cov)
        tempo_cov_reshaped_to_matrix[:, :] = cov_reshaped_to_matrix[:, :]
        tempo_cov_reshaped_to_matrix[0, 0] = 0
        tempo_cov_reshaped_to_matrix[1, 1] = 0
        tempo_cov_reshaped_to_matrix[2, 2] = 0
        tempo_cov_reshaped_to_matrix[3, 3] = 0
        print(np.max(np.abs(tempo_cov_reshaped_to_matrix)))
    # for j in range(cov_robustified.shape[1]):
    #     ax.plot(q_robustified[0, j*(polynomial_degree+1)], q_robustified[1, j*(polynomial_degree+1)], "ok", markersize=2)
    #     cov_reshaped_to_matrix = np.zeros(bio_model.matrix_shape_cov)
    #     for s_0 in range(bio_model.matrix_shape_cov[0]):
    #         for s_1 in range(bio_model.matrix_shape_cov[1]):
    #             cov_reshaped_to_matrix[s_0, s_1] = cov_robustified[bio_model.matrix_shape_cov[0] * s_1 + s_0, j]
    #     draw_cov_ellipse(cov_reshaped_to_matrix[:2, :2], q_robustified[:, j*(polynomial_degree + 1)], ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.axis("equal")
    # plt.legend()
    plt.savefig("output.png")
    plt.show()


if __name__ == "__main__":
    main()
