"""
This example presents how to implement wheel related holonomic and non-holonomic constraints in bioptim.
"""

import numpy as np
from casadi import vertcat

from bioptim import (
    BiMappingList,
    BoundsList,
    ConstraintList,
    DynamicsOptions,
    DynamicsOptionsList,
    HolonomicConstraintsFcn,
    HolonomicConstraintsList,
    TorqueBiorbdModel,
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    DynamicsEvaluation,
    DynamicsFunctions,
)
from bioptim.examples.utils import ExampleUtils



class ModifiedTorqueBiorbdModel(TorqueBiorbdModel):
    def __init__(
        self,
        bio_model_path: str,
    ):
        super().__init__(
            bio_model_path,
        )

    def dynamics(
        self,
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ):

        # Get states indices
        q_indices, qdot_indices = self.get_q_qdot_indices(nlp)

        # Get variables
        q, qdot, tau, external_forces = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )

        # Initialize dxdt
        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[q_indices, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dxdt[qdot_indices, 0] = DynamicsFunctions.compute_qddot(nlp, q, qdot, tau, external_forces)

        if nlp.model.fatigue is not None and "tau" in nlp.model.fatigue:
            dxdt = nlp.model.fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            DynamicsFunctions.no_states_mapping(nlp)
            slope_q, slope_qdot = self.get_basic_slopes(nlp)

            # Initialize defects
            defects = nlp.cx(nlp.states.shape, 1)

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:

                dxdt_defects = nlp.cx(nlp.states.shape, 1)
                dxdt_defects[q_indices, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
                dxdt_defects[qdot_indices, 0] = DynamicsFunctions.forward_dynamics(
                    nlp, q, qdot, tau, nlp.model.contact_types, external_forces
                )

                slopes = nlp.cx(nlp.states.shape, 1)
                slopes[q_indices, 0] = slope_q
                slopes[qdot_indices, 0] = slope_qdot

                # Get fatigue defects
                dxdt_defects, slopes = DynamicsFunctions.get_fatigue_defects(
                    "tau",
                    dxdt_defects,
                    slopes,
                    nlp,
                    states,
                    controls,
                )

                defects = slopes - dxdt_defects

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                if nlp.model.fatigue is not None:
                    raise NotImplementedError("Fatigue is not implemented yet with inverse dynamics defects.")

                defects[q_indices, 0] = slope_q - qdot

                tau_id = DynamicsFunctions.inverse_dynamics(
                    nlp,
                    q=q,
                    qdot=qdot,
                    qddot=slope_qdot,
                    contact_types=nlp.model.contact_types,
                    external_forces=external_forces,
                )
                tau_defects = tau - tau_id
                defects[qdot_indices, 0] = tau_defects
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for torque driven dynamics."
                )

            defects = vertcat(defects, DynamicsFunctions.get_contact_defects(nlp, q, qdot, slope_qdot))

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 30,
    final_time: float = 1,
    expand_dynamics: bool = False,
):
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """
    # # Create a holonomic constraint to create a double pendulum from two single pendulums
    # holonomic_constraints = HolonomicConstraintsList()
    # holonomic_constraints.add(
    #     "holonomic_constraints",
    #     HolonomicConstraintsFcn.superimpose_markers,
    #     marker_1="marker_1",
    #     marker_2="marker_3",
    #     index=slice(0, 3),
    #     local_frame_index=0,
    # )

    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint
    bio_model = HolonomicTorqueBiorbdModel(
        biorbd_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[1],
        dependent_joint_index=[0, 2, 3],
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", multi_thread=False)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), expand_dynamics=expand_dynamics))

    # Path Constraints
    constraints = ConstraintList()

    # Boundaries
    variable_bimapping = BiMappingList()
    # The rotations (joint 0 and 2) are independent. The rotations (joint 1 and 3) are constrained by the holonomic
    # constraint, so we need to map the states and controls to only compute the dynamics of the independent joints
    # The dynamics of the dependent joints will be computed from the holonomic constraint
    variable_bimapping.add("q", to_second=[None, 0, None, None], to_first=[1])
    variable_bimapping.add("qdot", to_second=[None, 0, None, None], to_first=[1])
    x_bounds = BoundsList()
    # q_u and qdot_u are the states of the independent joints
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=variable_bimapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=variable_bimapping)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q_u", [np.pi / 6])
    x_init.add("qdot_u", [0])
    x_bounds["q_u"][:, 0] = [np.pi / 6]
    x_bounds["qdot_u"][:, 0] = 0
    x_bounds["q_u"][:, -1] = [-np.pi / 4]
    # x_bounds["q_u"][1, -1] = 0

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    # # Only the rotations are controlled
    variable_bimapping.add("tau", to_second=[None, 0, None, None], to_first=[1])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min], max_bound=[tau_max])
    u_init = InitialGuessList()
    u_init.add("tau", [tau_init])

    # ------------- #

    return (
        OptimalControlProgram(
            bio_model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            variable_mappings=variable_bimapping,
            constraints=constraints,
        ),
        bio_model,
    )


def main():
    """
    Runs the optimization and animates it
    """

    model_path = ExampleUtils.folder + "/models/two_pendulums_rotule.bioMod"
    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    print(sol.real_time_to_optimize)

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = bio_model.compute_q_from_u_iterative(states["q_u"])

    viewer = "pyorerun"
    if viewer == "bioviz":
        import bioviz

        viz = bioviz.Viz(model_path)
        viz.load_movement(q)
        viz.exec()

    if viewer == "pyorerun":
        import pyorerun

        viz = pyorerun.PhaseRerun(t_span=np.concatenate(sol.decision_time()).squeeze())
        viz.add_animated_model(pyorerun.BiorbdModel(model_path), q=q)

        viz.rerun("double_pendulum")

    sol.graphs()


if __name__ == "__main__":
    main()
