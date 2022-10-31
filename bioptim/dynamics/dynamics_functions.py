from typing import Union

from casadi import horzcat, vertcat, MX, SX, Function

from ..misc.enums import RigidBodyDynamics
from .fatigue.fatigue_dynamics import FatigueList
from ..optimization.optimization_variable import OptimizationVariable
from ..optimization.non_linear_program import NonLinearProgram
from .dynamics_evaluation import DynamicsEvaluation


class DynamicsFunctions:
    """
    Implementation of all the dynamic functions

    Methods
    -------
    custom(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Interface to custom dynamic function provided by the user
    torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool)
        Forward dynamics driven by joint torques, optional external forces can be declared.
    torque_activations_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact) -> MX:
        Forward dynamics driven by joint torques activations.
    torque_derivative_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
        Forward dynamics driven by joint torques, optional external forces can be declared.
    forces_from_torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        Contact forces of a forward dynamics driven by joint torques with contact constraints.
    muscles_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
        Forward dynamics driven by muscle.
    forces_from_muscle_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.
    get(var: OptimizationVariable, cx: Union[MX, SX]):
        Main accessor to a variable in states or controls (cx)
    apply_parameters(parameters: MX.sym, nlp: NonLinearProgram)
        Apply the parameter variables to the model. This should be called before calling the dynamics
    compute_qdot(nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX]):
        Easy accessor to derivative of q
    forward_dynamics(nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX], tau: Union[MX, SX], with_contact: bool):
        Easy accessor to derivative of qdot
    compute_muscle_dot(nlp: NonLinearProgram, muscle_excitations: Union[MX, SX]):
        Easy accessor to derivative of muscle activations
    compute_tau_from_muscle(nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX], muscle_activations: Union[MX, SX]):
        Easy accessor to tau computed from muscles
    contact_forces(nlp: NonLinearProgram, q, qdot, tau):
        Easy accessor for the contact forces in contact dynamics
    """

    @staticmethod
    def custom(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> DynamicsEvaluation:
        """
        Interface to custom dynamic function provided by the user.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        MX.sym
            The defects of the implicit dynamics
        """

        return nlp.dynamics_type.dynamic_function(states, controls, parameters, nlp)

    @staticmethod
    def torque_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        with_contact: bool,
        rigidbody_dynamics: RigidBodyDynamics,
        fatigue: FatigueList,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques, optional external forces can be declared.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system
        with_contact: bool
            If the dynamic with contact should be used
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        fatigue : FatigueList
            A list of fatigue elements

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q_scaled"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot_scaled"], states)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        tau = DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue)

        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
        ):
            dxdt = MX(nlp.states.shape, 1)
            dxdt[nlp.states["q_scaled"].index, :] = dq
            dxdt[nlp.states["qdot_scaled"].index, :] = DynamicsFunctions.get(nlp.controls["qddot_scaled"], controls)
        elif (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
        ):
            dxdt = MX(nlp.states.shape, 1)
            dxdt[nlp.states["q_scaled"].index, :] = dq
            qddot = DynamicsFunctions.get(nlp.states["qddot_scaled"], states)
            dxdt[nlp.states["qdot_scaled"].index, :] = qddot
            dxdt[nlp.states["qddot_scaled"].index, :] = DynamicsFunctions.get(nlp.controls["qdddot_scaled"], controls)
        else:
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)
            dxdt = MX(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q_scaled"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot_scaled"].index, :] = ddq

        if fatigue is not None and "tau_scaled" in fatigue:
            dxdt = fatigue["tau_scaled"].dynamics(dxdt, nlp, states, controls)

        defects = None
        # todo: contacts and fatigue to be handled with implicit dynamics
        if not with_contact and fatigue is None:
            qddot = DynamicsFunctions.get(nlp.states_dot["qddot_scaled"], nlp.states_dot.mx_reduced)
            tau_id = DynamicsFunctions.inverse_dynamics(nlp, q, qdot, qddot, with_contact)
            defects = MX(dq.shape[0] + tau_id.shape[0], tau_id.shape[1])

            dq_defects = []
            for _ in range(tau_id.shape[1]):
                dq_defects.append(
                    dq
                    - DynamicsFunctions.compute_qdot(
                        nlp, q, DynamicsFunctions.get(nlp.states_dot["qdot_scaled"], nlp.states_dot.mx_reduced)
                    )
                )
            defects[: dq.shape[0], :] = horzcat(*dq_defects)
            # We modified on purpose the size of the tau to keep the zero in the defects in order to respect the dynamics
            defects[dq.shape[0] :, :] = tau - tau_id

        return DynamicsEvaluation(dxdt, defects)

    @staticmethod
    def __get_fatigable_tau(nlp: NonLinearProgram, states: MX, controls: MX, fatigue: FatigueList) -> MX:
        """
        Apply the forward dynamics including (or not) the torque fatigue

        Parameters
        ----------
        nlp: NonLinearProgram
            The current phase
        states: MX
            The states variable that may contains the tau and the tau fatigue variables
        controls: MX
            The controls variable that may contains the tau
        fatigue: FatigueList
            The dynamics for the torque fatigue

        Returns
        -------
        The generalized accelerations
        """

        tau_var, tau_mx = (nlp.controls, controls) if "tau_scaled" in nlp.controls else (nlp.states, states)
        tau = DynamicsFunctions.get(tau_var["tau_scaled"], tau_mx)
        if fatigue is not None and "tau_scaled" in fatigue:
            tau_fatigue = fatigue["tau_scaled"]
            tau_suffix = fatigue["tau_scaled"].suffix

            # Only homogeneous state_only is implemented yet
            n_state_only = sum([t.models.state_only for t in tau_fatigue])
            if 0 < n_state_only < len(fatigue["tau_scaled"]):
                raise NotImplementedError("fatigue list without homogeneous state_only flag is not supported yet")
            apply_to_joint_dynamics = sum([t.models.apply_to_joint_dynamics for t in tau_fatigue])
            if 0 < n_state_only < len(fatigue["tau_scaled"]):
                raise NotImplementedError(
                    "fatigue list without homogeneous apply_to_joint_dynamics flag is not supported yet"
                )
            if apply_to_joint_dynamics != 0:
                raise NotImplementedError("apply_to_joint_dynamics is not implemented for joint torque")

            if not tau_fatigue[0].models.split_controls and "tau_scaled" in nlp.controls:
                pass
            elif tau_fatigue[0].models.state_only:
                tau = sum([DynamicsFunctions.get(tau_var[f"tau_scaled_{suffix}"], tau_mx) for suffix in tau_suffix])
            else:
                tau = MX()
                for i, t in enumerate(tau_fatigue):
                    tau_tp = MX(1, 1)
                    for suffix in tau_suffix:
                        model = t.models.models[suffix]
                        tau_tp += (
                            DynamicsFunctions.get(nlp.states[f"tau_scaled_{suffix}_{model.dynamics_suffix()}"], states)[i]
                            * model.scaling
                        )
                    tau = vertcat(tau, tau_tp)
        return tau

    @staticmethod
    def torque_activations_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact):
        """
        Forward dynamics driven by joint torques activations.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system
        with_contact: bool
            If the dynamic with contact should be used

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q_scaled"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot_scaled"], states)
        tau_activations = DynamicsFunctions.get(nlp.controls["tau_scaled"], controls)

        tau = nlp.model.torque(tau_activations, q, qdot).to_mx()
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)

        dq = horzcat(*[dq for _ in range(ddq.shape[1])])

        return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)

    @staticmethod
    def torque_derivative_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        rigidbody_dynamics: RigidBodyDynamics,
        with_contact: bool,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques, optional external forces can be declared.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        with_contact: bool
            If the dynamic with contact should be used

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q_scaled"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot_scaled"], states)
        tau = DynamicsFunctions.get(nlp.states["tau_scaled"], states)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dtau = DynamicsFunctions.get(nlp.controls["taudot_scaled"], controls)

        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
        ):
            ddq = DynamicsFunctions.get(nlp.states["qddot_scaled"], states)
            dddq = DynamicsFunctions.get(nlp.controls["qdddot_scaled"], controls)

            dxdt = MX(nlp.states.shape, 1)
            dxdt[nlp.states["q_scaled"].index, :] = dq
            dxdt[nlp.states["qdot_scaled"].index, :] = ddq
            dxdt[nlp.states["qddot_scaled"].index, :] = dddq
            dxdt[nlp.states["tau_scaled"].index, :] = dtau
        else:
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)
            dxdt = MX(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q_scaled"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot_scaled"].index, :] = ddq
            dxdt[nlp.states["tau_scaled"].index, :] = horzcat(*[dtau for _ in range(ddq.shape[1])])

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    @staticmethod
    def forces_from_torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        """
        Contact forces of a forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)

        q_nlp, q_var = (nlp.states["q_scaled"], states) if "q_scaled" in nlp.states else (nlp.controls["q_scaled"], controls)
        qdot_nlp, qdot_var = (nlp.states["qdot_scaled"], states) if "qdot_scaled" in nlp.states else (nlp.controls["qdot_scaled"], controls)
        tau_nlp, tau_var = (nlp.states["tau_scaled"], states) if "tau_scaled" in nlp.states else (nlp.controls["tau_scaled"], controls)

        q = DynamicsFunctions.get(q_nlp, q_var)
        qdot = DynamicsFunctions.get(qdot_nlp, qdot_var)
        tau = DynamicsFunctions.get(tau_nlp, tau_var)

        return DynamicsFunctions.contact_forces(nlp, q, qdot, tau)

    @staticmethod
    def forces_from_torque_activation_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        """
        Contact forces of a forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)

        q_nlp, q_var = (nlp.states["q_scaled"], states) if "q_scaled" in nlp.states else (nlp.controls["q_scaled"], controls)
        qdot_nlp, qdot_var = (nlp.states["qdot_scaled"], states) if "qdot_scaled" in nlp.states else (nlp.controls["qdot_scaled"], controls)
        tau_nlp, tau_var = (nlp.states["tau_scaled"], states) if "tau_scaled" in nlp.states else (nlp.controls["tau_scaled"], controls)
        q = DynamicsFunctions.get(q_nlp, q_var)
        qdot = DynamicsFunctions.get(qdot_nlp, qdot_var)
        tau_activations = DynamicsFunctions.get(tau_nlp, tau_var)
        tau = nlp.model.torque(tau_activations, q, qdot).to_mx()

        return DynamicsFunctions.contact_forces(nlp, q, qdot, tau)

    @staticmethod
    def muscles_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        with_contact: bool,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        with_torque: bool = False,
        fatigue=None,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by muscle.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system
        with_contact: bool
            If the dynamic with contact should be used
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        fatigue: FatigueDynamicsList
            To define fatigue elements
        with_torque: bool
            If the dynamic should be added with residual torques

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q_scaled"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot_scaled"], states)
        residual_tau = DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue) if with_torque else None

        mus_act_nlp, mus_act = (nlp.states, states) if "muscles_scaled" in nlp.states else (nlp.controls, controls)
        mus_activations = DynamicsFunctions.get(mus_act_nlp["muscles_scaled"], mus_act)
        fatigue_states = None
        if fatigue is not None and "muscles_scaled" in fatigue:
            mus_fatigue = fatigue["muscles_scaled"]
            fatigue_name = mus_fatigue.suffix[0]

            # Sanity check
            n_state_only = sum([m.models.state_only for m in mus_fatigue])
            if 0 < n_state_only < len(fatigue["muscles_scaled"]):
                raise NotImplementedError(
                    f"{fatigue_name} list without homogeneous state_only flag is not supported yet"
                )
            apply_to_joint_dynamics = sum([m.models.apply_to_joint_dynamics for m in mus_fatigue])
            if 0 < apply_to_joint_dynamics < len(fatigue["muscles_scaled"]):
                raise NotImplementedError(
                    f"{fatigue_name} list without homogeneous apply_to_joint_dynamics flag is not supported yet"
                )

            dyn_suffix = mus_fatigue[0].models.models[fatigue_name].dynamics_suffix()
            fatigue_suffix = mus_fatigue[0].models.models[fatigue_name].fatigue_suffix()
            for m in mus_fatigue:
                for key in m.models.models:
                    if (
                        m.models.models[key].dynamics_suffix() != dyn_suffix
                        or m.models.models[key].fatigue_suffix() != fatigue_suffix
                    ):
                        raise ValueError(f"{fatigue_name} must be of all same types")

            if n_state_only == 0:
                mus_activations = DynamicsFunctions.get(nlp.states[f"muscles_scaled_{dyn_suffix}"], states)

            if apply_to_joint_dynamics > 0:
                fatigue_states = DynamicsFunctions.get(nlp.states[f"muscles_scaled_{fatigue_suffix}"], states)
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, fatigue_states)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            ddq = DynamicsFunctions.get(nlp.controls["qddot_scaled"], controls)
            dxdt = MX(nlp.states.shape, 1)
            dxdt[nlp.states["q_scaled"].index, :] = dq
            dxdt[nlp.states["qdot_scaled"].index, :] = DynamicsFunctions.get(nlp.controls["qddot_scaled"], controls)
        else:
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)
            dxdt = MX(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q_scaled"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot_scaled"].index, :] = ddq

        has_excitation = True if "muscles_scaled" in nlp.states else False
        if has_excitation:
            mus_excitations = DynamicsFunctions.get(nlp.controls["muscles_scaled"], controls)
            dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations)
            dxdt[nlp.states["muscles_scaled"].index, :] = horzcat(*[dmus for _ in range(ddq.shape[1])])

        if fatigue is not None and "muscles_scaled" in fatigue:
            dxdt = fatigue["muscles_scaled"].dynamics(dxdt, nlp, states, controls)

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    @staticmethod
    def forces_from_muscle_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        """
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q_scaled"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot_scaled"], states)
        residual_tau = DynamicsFunctions.get(nlp.controls["tau_scaled"], controls) if "tau_scaled" in nlp.controls else None

        mus_act_nlp, mus_act = (nlp.states, states) if "muscles_scaled" in nlp.states else (nlp.controls, controls)
        mus_activations = DynamicsFunctions.get(mus_act_nlp["muscles_scaled"], mus_act)
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        return DynamicsFunctions.contact_forces(nlp, q, qdot, tau)

    @staticmethod
    def joints_acceleration_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joints accelerations of a free floating body.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system
        rigidbody_dynamics: RigidBodyDynamics
            which rigid body dynamics to use

        Returns
        ----------
        MX.sym
            The derivative of states
        """
        if rigidbody_dynamics != RigidBodyDynamics.ODE:
            raise NotImplementedError("Implicit dynamics not implemented yet.")

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q_scaled"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot_scaled"], states)
        qddot_joints = DynamicsFunctions.get(nlp.controls["qddot_joints"], controls)

        qddot_root = nlp.model.ForwardDynamicsFreeFloatingBase(q, qdot, qddot_joints).to_mx()
        qddot_root_func = Function("qddot_root_func", [q, qdot, qddot_joints], [qddot_root]).expand()

        return DynamicsEvaluation(
            dxdt=vertcat(qdot, qddot_root_func(q, qdot, qddot_joints), qddot_joints), defects=None
        )

    @staticmethod
    def get(var: OptimizationVariable, cx: Union[MX, SX]):
        """
        Main accessor to a variable in states or controls (cx)

        Parameters
        ----------
        var: OptimizationVariable
            The variable from nlp.states["name"] or nlp.controls["name"]
        cx: Union[MX, SX]
            The actual SX or MX variables

        Returns
        -------
        The sliced values
        """

        return var.mapping.to_second.map(cx[var.index, :])

    @staticmethod
    def apply_parameters(parameters: MX.sym, nlp):
        """
        Apply the parameter variables to the model. This should be called before calling the dynamics

        Parameters
        ----------
        parameters: MX.sym
            The state of the system
        nlp: NonLinearProgram
            The definition of the system
        """

        offset = 0
        for param in nlp.parameters:
            # Call the pre dynamics function
            if param.function:
                param.function(nlp.model, parameters[offset : offset + param.size], **param.params)
                offset += param.size

    @staticmethod
    def compute_qdot(nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX]):
        """
        Easy accessor to derivative of q

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: Union[MX, SX]
            The value of q from "get"
        qdot: Union[MX, SX]
            The value of qdot from "get"

        Returns
        -------
        The derivative of q
        """

        q_nlp = nlp.states["q_scaled"] if "q_scaled" in nlp.states else nlp.controls["q_scaled"]
        return q_nlp.mapping.to_first.map(nlp.model.computeQdot(q, qdot).to_mx())

    @staticmethod
    def forward_dynamics(
        nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX], tau: Union[MX, SX], with_contact: bool
    ):
        """
        Easy accessor to derivative of qdot

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: Union[MX, SX]
            The value of q from "get"
        qdot: Union[MX, SX]
            The value of qdot from "get"
        tau: Union[MX, SX]
            The value of tau from "get"
        with_contact: bool
            If the dynamics with contact should be used

        Returns
        -------
        The derivative of qdot
        """
        qdot_var = nlp.states["qdot_scaled"] if "qdot_scaled" in nlp.states else nlp.controls["qdot_scaled"]

        if nlp.external_forces:
            dxdt = MX(len(qdot_var.mapping.to_first), nlp.ns)
            for i, f_ext in enumerate(nlp.external_forces):
                if with_contact:
                    qddot = nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau, f_ext).to_mx()
                else:
                    qddot = nlp.model.ForwardDynamics(q, qdot, tau, f_ext).to_mx()
                dxdt[:, i] = qdot_var.mapping.to_first.map(qddot)
            return dxdt
        else:
            if with_contact:
                qddot = nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()
            else:
                qddot = nlp.model.ForwardDynamics(q, qdot, tau).to_mx()
            return qdot_var.mapping.to_first.map(qddot)

    @staticmethod
    def inverse_dynamics(
        nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX], qddot: Union[MX, SX], with_contact: bool
    ):
        """
        Easy accessor to torques from inverse dynamics

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: Union[MX, SX]
            The value of q from "get"
        qdot: Union[MX, SX]
            The value of qdot from "get"
        qddot: Union[MX, SX]
            The value of qddot from "get"
        with_contact: bool
            If the dynamics with contact should be used

        Returns
        -------
        Torques in tau
        """

        tau_var = nlp.states["tau_scaled"] if "tau_scaled" in nlp.states else nlp.controls["tau_scaled"]

        if nlp.external_forces:
            tau = MX(tau_var.mx.shape[0], nlp.ns)
            for i, f_ext in enumerate(nlp.external_forces):
                tau[:, i] = nlp.model.InverseDynamics(q, qdot, qddot, f_ext).to_mx()
        else:
            tau = nlp.model.InverseDynamics(q, qdot, qddot).to_mx()
        return tau  # We ignore on purpose the mapping to keep zeros in the defects of the dynamic.

    @staticmethod
    def compute_muscle_dot(nlp: NonLinearProgram, muscle_excitations: Union[MX, SX]):
        """
        Easy accessor to derivative of muscle activations

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        muscle_excitations: Union[MX, SX]
            The value of muscle_excitations from "get"

        Returns
        -------
        The derivative of muscle activations
        """

        muscles_states = nlp.model.stateSet()
        for k in range(len(nlp.controls["muscles_scaled"])):
            muscles_states[k].setExcitation(muscle_excitations[k])
        return nlp.model.activationDot(muscles_states).to_mx()

    @staticmethod
    def compute_tau_from_muscle(
        nlp: NonLinearProgram,
        q: Union[MX, SX],
        qdot: Union[MX, SX],
        muscle_activations: Union[MX, SX],
        fatigue_states: Union[MX, SX] = None,
    ):
        """
        Easy accessor to tau computed from muscles

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: Union[MX, SX]
            The value of q from "get"
        qdot: Union[MX, SX]
            The value of qdot from "get"
        muscle_activations: Union[MX, SX]
            The value of muscle_activations from "get"
        fatigue_states: Union[MX, SX]
            The states of fatigue

        Returns
        -------
        The generalized forces computed from the muscles
        """

        muscles_states = nlp.model.stateSet()
        for k in range(len(nlp.controls["muscles_scaled"])):
            if fatigue_states is not None:
                muscles_states[k].setActivation(muscle_activations[k] * (1 - fatigue_states[k]))
            else:
                muscles_states[k].setActivation(muscle_activations[k])
        return nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()

    @staticmethod
    def contact_forces(nlp: NonLinearProgram, q, qdot, tau):
        """
        Easy accessor for the contact forces in contact dynamics

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: Union[MX, SX]
            The value of q from "get"
        qdot: Union[MX, SX]
            The value of qdot from "get"
        tau: Union[MX, SX]
            The value of tau from "get"

        Returns
        -------
        The contact forces
        """

        cs = nlp.model.getConstraints()
        if nlp.external_forces:
            all_cs = MX()
            for i, f_ext in enumerate(nlp.external_forces):
                nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau, cs, f_ext).to_mx()
                all_cs = horzcat(all_cs, cs.getForce().to_mx())
            return all_cs
        else:
            nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau, cs).to_mx()
            return cs.getForce().to_mx()
