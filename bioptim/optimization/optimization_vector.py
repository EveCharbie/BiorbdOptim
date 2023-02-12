import numpy as np
from casadi import vertcat, horzcat, DM, MX, SX

from .parameters import ParameterList, Parameter
from ..limits.path_conditions import Bounds, InitialGuess, InitialGuessList, NoisedInitialGuess
from ..misc.enums import ControlType, InterpolationType
from ..dynamics.ode_solver import OdeSolver


class OptimizationVector:
    """
    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    parameters_in_list: ParameterList
        A list of all the parameters in the ocp
    x: MX, SX
        The optimization variable for the states
    x_bounds: list
        A list of state bounds for each phase
    x_init: list
        A list of states initial guesses for each phase
    n_all_x: int
        The number of states of all the phases
    n_phase_x: list
        The number of states per phases
    u: MX, SX
        The optimization variable for the controls
    u_bounds: list
        A list of control bounds for each phase
    u_init: list
        A list of control initial guesses for each phase
    n_all_u: int
        The number of controls of all the phases
    n_phase_u: list
        The number of controls per phases

    Methods
    -------
    vector(self)
        Format the x, u and p so they are in one nice (and useful) vector
    bounds(self)
        Format the x, u and p bounds so they are in one nice (and useful) vector
    init(self)
        Format the x, u and p init so they are in one nice (and useful) vector
    extract_phase_time(self, data: np.ndarray | DM) -> list
        Get the phase time. If time is optimized, the MX/SX values are replaced by their actual optimized time
    to_dictionaries(self, data: np.ndarray | DM) -> tuple
        Convert a vector of solution in an easy to use dictionary, where are the variables are given their proper names
    define_ocp_shooting_points(self)
        Declare all the casadi variables with the right size to be used during a specific phase
    define_ocp_bounds(self)
        Declare and parse the bounds for all the variables (v vector)
    define_ocp_initial_guess(self)
        Declare and parse the initial guesses for all the variables (v vector)
    add_parameter(self, param: Parameter)
        Add a parameter to the parameters pool
    """

    def __init__(self, ocp):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """

        self.ocp = ocp

        self.parameters_in_list = ParameterList()

        self.x_scaled: MX | SX | list = []
        self.x_bounds = []
        self.x_init = []
        self.n_all_x = 0
        self.n_phase_x = []

        self.u_scaled: MX | SX | list = []
        self.u_bounds = []
        self.u_init = []
        self.n_all_u = 0
        self.n_phase_u = []

        for _ in range(self.ocp.n_phases):
            self.x_scaled.append([])
            self.x_bounds.append(Bounds(interpolation=InterpolationType.CONSTANT))
            self.x_init.append(InitialGuess(interpolation=InterpolationType.CONSTANT))
            self.n_phase_x.append(0)

            self.u_scaled.append([])
            self.u_bounds.append(Bounds(interpolation=InterpolationType.CONSTANT))
            self.u_init.append(InitialGuess(interpolation=InterpolationType.CONSTANT))
            self.n_phase_u.append(0)

    @property
    def vector(self):
        """
        Format the x, u and p so they are in one nice (and useful) vector for casadi solve

        Returns
        -------
        The vector of all variables
        """

        x_scaled = []
        u_scaled = []
        for nlp in self.ocp.nlp:
            if nlp.states_phase_mapping_idx.index is not None:
                index_x = nlp.states_phase_mapping_idx.index
            else:
                index_x = list(range(nlp.states.shape))

            for k in range(nlp.ns + 1):
                for liberty in range(nlp.states.shape):
                    if nlp.states_phase_mapping_idx.phase == nlp.phase_idx:
                        x_scaled.append(self.x_scaled[nlp.phase_idx][liberty, k])
                # for k in range(nlp.ns + 1):
                #     x_scaled[nlp.phase_idx].append([])
                #     for liberty in range(len(self.x_scaled[nlp.phase_idx][k])):
                #         x_scaled[nlp.phase_idx][k].append(self.x_scaled[nlp.phase_idx][k][liberty])
                #     x_scaled[nlp.phase_idx][k] = vertcat(*x_scaled[nlp.phase_idx][k])
                # x_scaled[nlp.phase_idx] = vertcat(*x_scaled[nlp.phase_idx])
                    else:
                        if liberty not in index_x:
                            x_scaled.append(self.x_scaled[nlp.phase_idx][liberty, k])
                # for k in range(nlp.ns+1) :
                #     x_scaled[nlp.phase_idx].append([])
                #     for liberty in range(len(self.x_scaled[nlp.phase_idx][k])):
                #         if liberty not in index_x :
                #             x_scaled[nlp.phase_idx][k].append(self.x_scaled[nlp.phase_idx][k][liberty])
                #     x_scaled[nlp.phase_idx][k] = vertcat(*x_scaled[nlp.phase_idx][k])
                # x_scaled[nlp.phase_idx] = vertcat(*x_scaled[nlp.phase_idx])

            if nlp.controls_phase_mapping_idx.index is not None:
                index_u = nlp.controls_phase_mapping_idx.index
            else:
                index_u = list(range(nlp.controls.shape))

            for k in range(nlp.ns + 1):
                if nlp.control_type != ControlType.CONSTANT or (
                        nlp.control_type == ControlType.CONSTANT and k != nlp.ns):
                    for liberty in range(nlp.controls.shape):
                        if nlp.controls_phase_mapping_idx.phase == nlp.phase_idx:
                            u_scaled.append(self.u_scaled[nlp.phase_idx][liberty, k])
                        else:
                            if liberty not in index_u:
                                u_scaled.append(self.u_scaled[nlp.phase_idx][liberty, k])

            # if nlp.use_controls_from_phase_idx == nlp.phase_idx :
            #     u_scaled.append([])
            #     for k in range(nlp.ns):
            #         u_scaled[nlp.phase_idx].append([])
            #         for liberty in range(len(self.u_scaled[nlp.phase_idx][k])) :
            #             u_scaled[nlp.phase_idx][k].append(self.u_scaled[nlp.phase_idx][k][liberty])
            #         u_scaled[nlp.phase_idx][k] = vertcat(*u_scaled[nlp.phase_idx][k])
            #     u_scaled[nlp.phase_idx] = vertcat(*u_scaled[nlp.phase_idx])
            # else:
            #     if nlp.controls.shape != len(index_u):
            #         u_scaled.append([])
            #         for k in range(nlp.ns) :
            #             u_scaled[nlp.phase_idx].append([])
            #             if self.u_scaled[nlp.phase_idx][k] :
            #                 for liberty in range(len(self.u_scaled[nlp.phase_idx][k])):
            #                     if liberty not in index_u :
            #                         u_scaled[nlp.phase_idx][k].append(self.u_scaled[nlp.phase_idx][k][liberty])
            #                 u_scaled[nlp.phase_idx][k] = vertcat(*u_scaled[nlp.phase_idx][k])
            #         u_scaled[nlp.phase_idx] = vertcat(*u_scaled[nlp.phase_idx])

        ##### To be removed #####
        # Comments from L.Sechoir
       # for phase in range(len(self.x)) :
       #     for node in range(len(self.x[phase])) :
        #        for liberty in range(len(self.x[phase][node])) :
        #            x += [self.x[phase][node][liberty]]
       # for phase in range(len(self.u)) :
        #    for node in range(len(self.u[phase])) :
         #       for liberty in range(len(self.u[phase][node])) :
          #          u += [self.u[phase][node][liberty]]


        # si on veut toutes les variables avec doublons
       # x =[]

       # for phase in range(len(self.x)):
        #    for k in range(len(self.x[phase])) :
         #       x_1 = vertcat(*self.x[phase][k])
         #       x.append(x_1)
         #       if len(x) > 1 :
         #           vertcat(x[0],x[1] )

       # vertcat(*x)

            #if nlp.use_controls_from_phase_idx == nlp.phase_idx:
             #   u += [self.u[nlp.phase_idx]]
           # else:
              #  for liberty in len(nlp.controls.shape):
              #     if liberty in index_u:
                 #       u += [self.x[nlp.phase_idx][0: node_all][liberty]]

       # for i in len(self.ocp.nlp) :
        #    x = vertcat(x[0],self.ocp.nlp[i].X)
        #    x = vertcat(u[0], self.ocp.nlp[i].U)
        # lenght = 0
        # for x_i in x :
        #     lenght += x_i.shape
        # for u_i in u :
        #     lenght += u_i.shape
        # print(f"lenght is {lenght}")

        return vertcat(*x_scaled, *u_scaled, self.parameters_in_list.cx)

    @property
    def bounds(self):
        """
        Format the x, u and p bounds so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all bounds
        """

        if isinstance(self.ocp.nlp[0].ode_solver, OdeSolver.COLLOCATION) and not isinstance(
            self.ocp.nlp[0].ode_solver, OdeSolver.IRK
        ):
            n_steps = self.ocp.nlp[0].ode_solver.steps + 1
        else:
            n_steps = 1

        v_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        for phase, x_bound in enumerate(self.x_bounds):
            v_bounds.concatenate(
                x_bound.scale(self.ocp.nlp[phase].x_scaling["all"].to_vector(self.ocp.nlp[phase].ns * n_steps + 1))
            )

        for phase, u_bound in enumerate(self.u_bounds):
            if self.ocp.nlp[0].control_type == ControlType.LINEAR_CONTINUOUS:
                ns = self.ocp.nlp[phase].ns + 1
            else:
                ns = self.ocp.nlp[phase].ns
            v_bounds.concatenate(u_bound.scale(self.ocp.nlp[phase].u_scaling["all"].to_vector(ns)))

        for param in self.parameters_in_list:
            v_bounds.concatenate(param.bounds.scale(param.scaling))

        return v_bounds

    @property
    def init(self):
        """
        Format the x, u and p init so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all init
        """
        v_init = InitialGuess(interpolation=InterpolationType.CONSTANT)
        if isinstance(self.ocp.nlp[0].ode_solver, OdeSolver.COLLOCATION) and not isinstance(
            self.ocp.nlp[0].ode_solver, OdeSolver.IRK
        ):
            steps = self.ocp.nlp[0].ode_solver.steps + 1
        else:
            steps = 1

        for phase, x_init in enumerate(self.x_init):
            nlp = self.ocp.nlp[phase]

            if isinstance(self.ocp.original_values["x_init"], InitialGuessList):
                original_x_init = self.ocp.original_values["x_init"][phase]
            else:
                original_x_init = self.ocp.original_values["x_init"]
            interpolation_type = None if original_x_init is None else original_x_init.type

            if nlp.ode_solver.is_direct_collocation and interpolation_type == InterpolationType.EACH_FRAME:
                v_init.concatenate(
                    self._init_linear_interpolation(phase=phase).scale(
                        self.ocp.nlp[phase].x_scaling["all"].to_vector(self.ocp.nlp[phase].ns * steps + 1),
                    )
                )
            else:
                v_init.concatenate(
                    x_init.scale(self.ocp.nlp[phase].x_scaling["all"].to_vector(self.ocp.nlp[phase].ns * steps + 1))
                )

        for phase, u_init in enumerate(self.u_init):
            if self.ocp.nlp[0].control_type == ControlType.LINEAR_CONTINUOUS:
                ns = self.ocp.nlp[phase].ns + 1
            else:
                ns = self.ocp.nlp[phase].ns
            v_init.concatenate(u_init.scale(self.ocp.nlp[phase].u_scaling["all"].to_vector(ns)))

        for param in self.parameters_in_list:
            v_init.concatenate(param.initial_guess.scale(param.scaling))

        return v_init

    def _init_linear_interpolation(self, phase: int) -> InitialGuess:
        """
        Perform linear interpolation between shooting nodes so that initial guess values are defined for each
        collocation point

        Parameters
        ----------
        phase: int
            The phase index

        Returns
        -------
        The initial guess for the states variables for all collocation points

        """
        nlp = self.ocp.nlp[phase]
        n_points = nlp.ode_solver.polynomial_degree + 1
        x_init_vector = np.zeros((nlp.states["scaled"].shape, self.n_phase_x[phase] // nlp.states["scaled"].shape))
        init_values = (
            self.ocp.original_values["x_init"][phase].init
            if isinstance(self.ocp.original_values["x_init"], InitialGuessList)
            else self.ocp.original_values["x_init"].init
        )
        # the linear interpolation is performed at the given time steps from the ode solver
        steps = np.array(nlp.ode_solver.integrator(self.ocp, nlp)[0].step_time)

        for idx_state, state in enumerate(init_values):
            for frame in range(nlp.ns):
                x_init_vector[idx_state, frame * n_points : (frame + 1) * n_points] = (
                    state[frame] + (state[frame + 1] - state[frame]) * steps
                )

            x_init_vector[idx_state, -1] = state[nlp.ns]

        x_init_reshaped = x_init_vector.reshape((1, -1), order="F").T
        return InitialGuess(x_init_reshaped)

    def extract_phase_time(self, data: np.ndarray | DM) -> list:
        """
        Get the phase time. If time is optimized, the MX/SX values are replaced by their actual optimized time

        Parameters
        ----------
        data: np.ndarray | DM
            The solution in a vector

        Returns
        -------
        The phase time
        """

        offset = self.n_all_x + self.n_all_u
        data_time_optimized = []
        if "time" in self.parameters_in_list.names:
            for param in self.parameters_in_list:
                if param.name == "time":
                    data_time_optimized = list(np.array(data[offset : offset + param.size])[:, 0])
                    break
                offset += param.size

        phase_time = [0] + [nlp.tf for nlp in self.ocp.nlp]
        if data_time_optimized:
            cmp = 0
            for i in range(len(phase_time)):
                if isinstance(phase_time[i], self.ocp.cx):
                    phase_time[i] = data_time_optimized[self.ocp.parameter_mappings["time"].to_second.map_idx[cmp]]
                    cmp += 1
        return phase_time

    def to_dictionaries(self, data: np.ndarray | DM) -> tuple:
        """
        Convert a vector of solution in an easy to use dictionary, where are the variables are given their proper names

        Parameters
        ----------
        data: np.ndarray | DM
            The solution in a vector

        Returns
        -------
        The solution in a tuple of dictionaries format (tuple => each phase)
        """

        ocp = self.ocp
        v_array = np.array(data).squeeze()

        data_states = []
        data_controls = []
        for _ in range(self.ocp.n_phases):
            data_states.append({})
            data_controls.append({})
        data_parameters = {}

        offset = 0
        p_idx = 0
        for p in range(self.ocp.n_phases):
            if self.ocp.nlp[p].use_states_from_phase_idx == self.ocp.nlp[p].phase_idx:
                x_array = v_array[offset : offset + self.n_phase_x[p]].reshape(
                    (ocp.nlp[p].states["scaled"].shape, -1), order="F"
                )
                data_states[p_idx]["all"] = x_array
                offset_var = 0
                for var in ocp.nlp[p].states["scaled"]:
                    data_states[p_idx][var] = x_array[
                        offset_var : offset_var + len(ocp.nlp[p].states["scaled"][var]), :
                    ]
                    offset_var += len(ocp.nlp[p].states["scaled"][var])
                p_idx += 1
                offset += self.n_phase_x[p]

        offset = self.n_all_x
        p_idx = 0
        for p in range(self.ocp.n_phases):
            if self.ocp.nlp[p].use_controls_from_phase_idx == self.ocp.nlp[p].phase_idx:
                u_array = v_array[offset : offset + self.n_phase_u[p]].reshape(
                    (ocp.nlp[p].controls["scaled"].shape, -1), order="F"
                )
                data_controls[p_idx]["all"] = u_array
                offset_var = 0
                for var in ocp.nlp[p].controls["scaled"]:
                    data_controls[p_idx][var] = u_array[
                        offset_var : offset_var + len(ocp.nlp[p].controls["scaled"][var]), :
                    ]
                    offset_var += len(ocp.nlp[p].controls["scaled"][var])
                p_idx += 1
                offset += self.n_phase_u[p]

        offset = self.n_all_x + self.n_all_u
        scaling_offset = 0
        data_parameters["all"] = v_array[offset:, np.newaxis] * ocp.nlp[0].parameters.scaling
        if len(data_parameters["all"].shape) == 1:
            data_parameters["all"] = data_parameters["all"][:, np.newaxis]
        for param in self.parameters_in_list:
            data_parameters[param.name] = v_array[offset : offset + param.size, np.newaxis] * param.scaling
            offset += param.size
            scaling_offset += param.size
            if len(data_parameters[param.name].shape) == 1:
                data_parameters[param.name] = data_parameters[param.name][:, np.newaxis]

        return data_states, data_controls, data_parameters

    def define_ocp_shooting_points(self):
        """
        Declare all the casadi variables with the right size to be used during a specific phase
        """

        x = []
        x_scaled = []
        u = []
        u_scaled = []
        for nlp in self.ocp.nlp:
            if nlp.states_phase_mapping_idx.index is not None:
                index_x = nlp.states_phase_mapping_idx.index
            else:
                index_x = list(range(nlp.states.shape))
            if nlp.controls_phase_mapping_idx.index is not None:
                index_u = nlp.controls_phase_mapping_idx.index
            else:
                index_u = list(range(nlp.controls.shape))
            x.append([])
            x_scaled.append([])
            u.append([])
            u_scaled.append([])
            if nlp.control_type != ControlType.CONSTANT and nlp.control_type != ControlType.LINEAR_CONTINUOUS:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            # phase, node, var
            for k in range(nlp.ns + 1):
                x[nlp.phase_idx].append([])
                x_scaled[nlp.phase_idx].append([])
                # states
                if nlp.phase_idx == nlp.states_phase_mapping_idx.phase:
                    for liberty in range(nlp.states.shape):
                        if k != nlp.ns and nlp.ode_solver.is_direct_collocation:
                            x_scaled[nlp.phase_idx][k].append(
                                nlp.cx.sym(
                                    "X_scaled" + str(nlp.phase_idx) + "_" + str(k) + '_' + str(liberty),
                                    1,
                                    nlp.ode_solver.polynomial_degree + 1
                                )
                            )
                            x[nlp.phase_idx][k].append(x_scaled[nlp.phase_idx][k][liberty] * nlp.x_scaling["all"].scaling[liberty])
                        else:
                            x_scaled[nlp.phase_idx][k].append(
                                nlp.cx.sym(
                                    "X_scaled_" + str(nlp.phase_idx) + "_" + str(k) + "_" + str(liberty),
                                    1,
                                    1)
                            )
                            x[nlp.phase_idx][k].append(x_scaled[nlp.phase_idx][k][liberty] * nlp.x_scaling["all"].scaling[liberty])
                else:
                    for liberty in range(nlp.states.shape):
                        if liberty in index_x:
                            x_scaled[nlp.phase_idx][k].append(x_scaled[nlp.states_phase_mapping_idx.phase][k][liberty])
                            x[nlp.phase_idx][k].append(x[nlp.states_phase_mapping_idx.phase][k][liberty])
                        else:
                            if k != nlp.ns and nlp.ode_solver.is_direct_collocation:
                                x_scaled[nlp.phase_idx][k].append(
                                    nlp.cx.sym(
                                        "X_scaled" + str(nlp.phase_idx) + "_" + str(k) + '_' + str(liberty),
                                        1,
                                        nlp.ode_solver.polynomial_degree + 1,
                                    )
                                )
                                x[nlp.phase_idx][k].append(
                                    x_scaled[nlp.phase_idx][k][liberty] * nlp.x_scaling["all"].scaling[liberty])
                            else:
                                x_scaled[nlp.phase_idx][k].append(
                                    nlp.cx.sym("X_scaled" + str(nlp.phase_idx) + "_" + str(k) + '_' + str(liberty), 1, 1)
                                )
                                x[nlp.phase_idx][k].append(x_scaled[nlp.phase_idx][k][liberty] * nlp.x_scaling["all"].scaling[liberty])
                #controls
                u[nlp.phase_idx].append([])
                u_scaled[nlp.phase_idx].append([])
                if nlp.phase_idx == nlp.controls_phase_mapping_idx.phase:
                    for liberty in range(nlp.controls.shape):
                        if nlp.control_type != ControlType.CONSTANT or (nlp.control_type == ControlType.CONSTANT and k != nlp.ns):
                            u_scaled[nlp.phase_idx][k].append(nlp.cx.sym("U_scaled" + str(nlp.phase_idx) + "_" + str(k) + '_' + str(liberty), 1, 1))
                            u[nlp.phase_idx][k].append(u_scaled[nlp.phase_idx][k][liberty] * nlp.u_scaling["all"].scaling[liberty])
                else:
                    for liberty in range(nlp.controls.shape):
                        if nlp.control_type != ControlType.CONSTANT or (
                                nlp.control_type == ControlType.CONSTANT and k != nlp.ns):
                            if liberty in index_u:
                                u_scaled[nlp.phase_idx][k].append(u_scaled[nlp.controls_phase_mapping_idx.phase][k][liberty])
                                u[nlp.phase_idx][k].append(u[nlp.controls_phase_mapping_idx.phase][k][liberty])
                            else:
                                u_scaled[nlp.phase_idx][k].append(nlp.cx.sym("U_scaled" + str(nlp.phase_idx) + "_" + str(k) + '_' + str(liberty), 1, 1))
                                u[nlp.phase_idx][k].append(u_scaled[nlp.phase_idx][k][liberty] * nlp.u_scaling["all"].scaling[liberty])

            # nlp.X_scaled = vertcat(*x_scaled[nlp.phase_idx])
            # nlp.X = vertcat(*x[nlp.phase_idx])
            nlp.X_scaled = x_scaled[nlp.phase_idx]
            nlp.X = x[nlp.phase_idx]
            for i, x_tp in enumerate(x_scaled[nlp.phase_idx]):
                if i == 0:
                    x_scaled_all = vertcat(*x_scaled[nlp.phase_idx][0])
                else:
                    x_scaled_all = horzcat(x_scaled_all, vertcat(*x_tp))
            self.x_scaled[nlp.phase_idx] = x_scaled_all
            if nlp.states_phase_mapping_idx.index is not None:
                self.n_phase_x[nlp.phase_idx] = self.x_scaled[nlp.phase_idx].size()[0] - len(nlp.states_phase_mapping_idx.index)*(nlp.ns+1)
            else:
                self.n_phase_x[nlp.phase_idx] = self.x_scaled[nlp.phase_idx].size()[0]
            # nlp.U_scaled = vertcat(*u_scaled[nlp.phase_idx])
            # nlp.U = vertcat(*u[nlp.phase_idx])
            nlp.U_scaled = u_scaled[nlp.phase_idx]
            nlp.U = u[nlp.phase_idx]
            for i, u_tp in enumerate(u_scaled[nlp.phase_idx]):
                if i == 0:
                    u_scaled_all = vertcat(*u_scaled[nlp.phase_idx][0])
                else:
                    u_scaled_all = horzcat(u_scaled_all, vertcat(*u_tp))
            self.u_scaled[nlp.phase_idx] = u_scaled_all
            if nlp.controls_phase_mapping_idx.index is not None:
                self.n_phase_u[nlp.phase_idx] = self.u_scaled[nlp.phase_idx].size()[0] - len(nlp.controls_phase_mapping_idx.index)*(nlp.ns+1 if nlp.control_type != ControlType.CONSTANT else nlp.ns)
            else:
                self.n_phase_u[nlp.phase_idx] = self.u_scaled[nlp.phase_idx].size()[0]

      ##### Comments for L.Sechoire #####
      # To be removed
                                   #if liberty > 0:
                                     #   vertcat(u[nlp.phase_idx][0], u[nlp.phase_idx][1])
                            # else:
                                  #  u[nlp.phase_idx][k].append(
                                   #     nlp.cx.sym("U_" + str(nlp.phase_idx) + "_" + str(k) + '_' + str(liberty), 1, 1)
                                  #  )
                                  #  if len(index_u) != nlp.controls.shape:
                                       # self.u[nlp.phase_idx][k].append(u[nlp.phase_idx][k][liberty])
                                  #  self.u[nlp.phase_idx][k] = vertcat(self.u[nlp.phase_idx][k][0],self.u[nlp.phase_idx][k][1])
                                    #if liberty > 0:
                                     #   vertcat(u[nlp.phase_idx][k][0], u[nlp.phase_idx][k][1])

                     #   self.u[nlp.phase_idx][k] = vertcat(*self.u[nlp.phase_idx][k])

                    #u[nlp.phase_idx][k] = vertcat(*[z for z in u[nlp.phase_idx][k]])
                #x[nlp.phase_idx][k] = vertcat(*[z for z in x[nlp.phase_idx][k]]) # vertcat(*x[nlp.phase_idx][k])

           # = u[nlp.phase_idx]
            #self.x[nlp.phase_idx] = x[nlp.phase_idx]  # pas de vertcat car on a besoin des indices
        #


        # for nlp in self.ocp.nlp:
        #     for k in range(nlp.ns + 1):  # k -> k
        #         self.x_scaled[nlp.phase_idx].append([])
        #         for liberty in range(len(x_scaled[nlp.phase_idx][k])):
        #             #if freedom not in index_x :
        #             self.x_scaled[nlp.phase_idx][k].append(x_scaled[nlp.phase_idx][k][liberty])
        #         x_scaled[nlp.phase_idx][k] = vertcat(*x_scaled[nlp.phase_idx][k])
        #
        #     nlp.X_scaled = x_scaled[nlp.phase_idx]
        #     nlp.X = x[nlp.phase_idx]
        #     self.x_scaled[nlp.phase_idx] = vertcat(
        #         *[x_tp.reshape((-1, 1)) for x_tp in x_scaled[nlp.use_states_from_phase_idx]]
        #     )
        #     for k in range(nlp.ns + 1):
        #         self.u_scaled[nlp.phase_idx].append([])
        #         for liberty in range(len(u_scaled[nlp.phase_idx][k])):
        #            # if liberty not in index_u :
        #             self.u_scaled[nlp.phase_idx][k].append(u_scaled[nlp.phase_idx][k][liberty])
        #         u_scaled[nlp.phase_idx][k] = vertcat(*u_scaled[nlp.phase_idx][k])
        #
        #     nlp.U_scaled = u_scaled[nlp.phase_idx]
        #     nlp.U = u[nlp.phase_idx]
        #     self.u_scaled[nlp.phase_idx] = vertcat(*u_scaled[nlp.use_controls_from_phase_idx])
        #
        #     self.n_phase_u[nlp.phase_idx] = (
        #         len(self.u_scaled[nlp.phase_idx]) * len(self.u_scaled[nlp.phase_idx][0])
        #         if nlp.phase_idx == nlp.use_controls_from_phase_idx else (nlp.controls.shape - len(index_u))*nlp.ns
        #     )

            ##### Comments from L.Sechoire #####
            # To be removed

            #self.x[nlp.phase_idx] = vertcat(*self.x[nlp.phase_idx])
            #self.u[nlp.phase_idx] = vertcat(*self.u[nlp.phase_idx])
              #  x_tp_all = MX()
            #self.x[nlp.phase_idx] = vertcat(*[x_tp.reshape((-1, 1)) for x_tp in x[nlp.use_states_from_phase_idx]])


              #  for x_tp in x[nlp.phase_idx]: # a vérifier
               #     x_tp_all = vertcat(x_tp_all, x_tp)
               # self.x[nlp.phase_idx] = x_tp_all


            #self.u[nlp.phase_idx] = vertcat(*u[nlp.use_controls_from_phase_idx])


               # u_tp_all = MX()
                #for u_tp in u[nlp.phase_idx]:
               #     u_tp_all = vertcat(u_tp_all, u_tp)
               # self.u[nlp.phase_idx] = u_tp_all


        self.n_all_x = sum(self.n_phase_x)
        self.n_all_u = sum(self.n_phase_u)

    def define_ocp_bounds(self):
        """
        Declare and parse the bounds for all the variables (v vector)
        """

        ocp = self.ocp

        # Sanity check
        for nlp in ocp.nlp:
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                nlp.x_bounds.check_and_adjust_dimensions(nlp.states.shape, nlp.ns)
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type == ControlType.CONSTANT:
                    nlp.u_bounds.check_and_adjust_dimensions(nlp.controls.shape, nlp.ns - 1)
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    nlp.u_bounds.check_and_adjust_dimensions(nlp.controls.shape, nlp.ns)
                else:
                    raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                nx = nlp.states.shape
                if nlp.ode_solver.is_direct_collocation:
                    all_nx = nx * nlp.ns * (nlp.ode_solver.polynomial_degree + 1) + nx
                    outer_offset = nx * (nlp.ode_solver.polynomial_degree + 1)
                    repeat = nlp.ode_solver.polynomial_degree + 1
                else:
                    all_nx = nx * (nlp.ns + 1)
                    outer_offset = nx
                    repeat = 1
                x_bounds = Bounds([0] * all_nx, [0] * all_nx, interpolation=InterpolationType.CONSTANT)
                for k in range(nlp.ns + 1):
                    for p in range(repeat if k != nlp.ns else 1):
                        span = slice(k * outer_offset + p * nx, k * outer_offset + (p + 1) * nx)
                        point = k if k != 0 else 0 if p == 0 else 1
                        x_bounds.min[span, 0] = nlp.x_bounds.min.evaluate_at(shooting_point=point)
                        x_bounds.max[span, 0] = nlp.x_bounds.max.evaluate_at(shooting_point=point)


            # Code from L.Sechoire
            # else :
            #     if nlp.index_x == [] :
            #         index_x = [i for i in range(nlp.states.shape)]
            #     nx = nlp.states.shape #nombre de degré de liberté
            #     index_roots = [i for i in range(nx) if i not in index_x]
            #     if nlp.ode_solver.is_direct_collocation:
            #         all_nx = nx * nlp.ns * (nlp.ode_solver.polynomial_degree + 1) + nx
            #         outer_offset = nx * (nlp.ode_solver.polynomial_degree + 1)
            #         repeat = nlp.ode_solver.polynomial_degree + 1
            #     else:
            #         all_nx = nx * (nlp.ns + 1)
            #         outer_offset = nx
            #         repeat = 1
            #     x_bounds = Bounds([0] * nx, [0] * nx, interpolation=InterpolationType.CONSTANT) # définit un une liste de 0 pour initialiser le vecteur
            #     for k in range(nlp.ns + 1):
            #         for p in range(repeat if k != nlp.ns else 1):
            #             #for index in index_x :
            #             a,b = [k * outer_offset + p * len(index_roots), k * outer_offset + (p + 1) * len(index_roots)]
            #             point = k if k != 0 else 0 if p == 0 else 1
            #             x_bounds.min[a:b, 0] = nlp.x_bounds.min.evaluate_at(shooting_point=point)[a:b]
            #             x_bounds.max[a:b, 0] = nlp.x_bounds.max.evaluate_at(shooting_point=point)[a:b]


                self.x_bounds[i_phase] = x_bounds

            # For controls
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type == ControlType.CONSTANT:
                    ns = nlp.ns
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    ns = nlp.ns + 1
                else:
                    raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
                nu = nlp.controls.shape
                all_nu = nu * ns
                u_bounds = Bounds([0] * all_nu, [0] * all_nu, interpolation=InterpolationType.CONSTANT)
                for k in range(ns):
                    u_bounds.min[k * nu : (k + 1) * nu, 0] = nlp.u_bounds.min.evaluate_at(shooting_point=k)
                    u_bounds.max[k * nu : (k + 1) * nu, 0] = nlp.u_bounds.max.evaluate_at(shooting_point=k)

            # code from L.Sechoire
            # else :
            #     if nlp.index_u == [] :
            #         index_u = [i for i in range(nlp.states.controls)]
            #     if nlp.control_type == ControlType.CONSTANT:
            #         ns = nlp.ns
            #     elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
            #         ns = nlp.ns + 1
            #     else:
            #         raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
            #     nu = nlp.controls.shape
            #     all_nu = (len(index_u)) * ns
            #     u_bounds = Bounds([0] * all_nu, [0] * all_nu, interpolation=InterpolationType.CONSTANT)
            #     for k in range(ns):
            #         u_bounds.min[k * nu : (k + 1) * nu, 0] = nlp.u_bounds.min.evaluate_at(shooting_point=k)
            #         u_bounds.max[k * nu : (k + 1) * nu, 0] = nlp.u_bounds.max.evaluate_at(shooting_point=k)

                self.u_bounds[i_phase] = u_bounds

    def get_ns(self, phase: int, interpolation_type: InterpolationType) -> int:
        """
        Define the number of shooting nodes and collocation points

        Parameters
        ----------
        phase: int
            The index of the current phase of the ocp
        interpolation_type: InterpolationType
            The interpolation type of x_init

        Returns
        -------
        ns: int
            The number of shooting nodes and collocation points
        """
        ocp = self.ocp
        ns = ocp.nlp[phase].ns
        if ocp.nlp[phase].ode_solver.is_direct_collocation:
            if interpolation_type != InterpolationType.EACH_FRAME:
                ns *= ocp.nlp[phase].ode_solver.steps + 1
        return ns

    def define_ocp_initial_guess(self):
        """
        Declare and parse the initial guesses for all the variables (v vector)
        """

        ocp = self.ocp
        # index_x = ocp.nlp[0].index_x
        # index_roots_x = [0,1,2,3,4,5,16,17,18,19,20,21] #indices des nouvelles variables /non mappées
        # index_roots_u = []
        # Sanity check
        for nlp in ocp.nlp:
            interpolation = nlp.x_init.type
            ns = self.get_ns(phase=nlp.phase_idx, interpolation_type=interpolation)
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                if nlp.ode_solver.is_direct_shooting:
                    if nlp.x_init.type == InterpolationType.ALL_POINTS:
                        raise ValueError("InterpolationType.ALL_POINTS must only be used with direct collocation")
                nlp.x_init.check_and_adjust_dimensions(nlp.states.shape, ns)

                nlp.x_init.check_and_adjust_dimensions(nlp.states.shape, ns)
            else :
                if nlp.ode_solver.is_direct_shooting:
                    if nlp.x_init.type == InterpolationType.ALL_POINTS:
                        raise ValueError("InterpolationType.ALL_POINTS must only be used with direct collocation")
                nlp.x_init.check_and_adjust_dimensions(nlp.states.shape, ns)
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type == ControlType.CONSTANT:
                    nlp.u_init.check_and_adjust_dimensions(nlp.controls.shape, nlp.ns - 1)
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    nlp.u_init.check_and_adjust_dimensions(nlp.controls.shape, nlp.ns)
                else:
                    raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")
            else :
                if nlp.control_type == ControlType.CONSTANT:
                    nlp.u_init.check_and_adjust_dimensions(nlp.controls.shape, nlp.ns - 1)
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    nlp.u_init.check_and_adjust_dimensions(nlp.controls.shape, nlp.ns)
                else:
                    raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                nx = nlp.states.shape
                if nlp.ode_solver.is_direct_collocation and nlp.x_init.type != InterpolationType.EACH_FRAME:
                    all_nx = nx * nlp.ns * (nlp.ode_solver.polynomial_degree + 1) + nx
                    outer_offset = nx * (nlp.ode_solver.polynomial_degree + 1)
                    repeat = nlp.ode_solver.polynomial_degree + 1
                else:
                    all_nx = nx * (nlp.ns + 1)
                    outer_offset = nx
                    repeat = 1

                x_init = InitialGuess([0] * all_nx, interpolation=InterpolationType.CONSTANT)
                for k in range(nlp.ns + 1):
                    for p in range(repeat if k != nlp.ns else 1):
                        span = slice(k * outer_offset + p * nx, k * outer_offset + (p + 1) * nx)
                        point = k if k != 0 else 0 if p == 0 else 1
                        if isinstance(nlp.x_init, NoisedInitialGuess):
                            if nlp.x_init.type == InterpolationType.ALL_POINTS:
                                point = k * repeat + p
                        elif isinstance(nlp.x_init, InitialGuess) and nlp.x_init.type == InterpolationType.EACH_FRAME:
                            point = k * repeat + p
                        x_init.init[span, 0] = nlp.x_init.init.evaluate_at(shooting_point=point)
                self.x_init[i_phase] = x_init

            # Code from L.Sechoire
            # else :
            #
            #     nx = len(index_roots_x)
            #     if nlp.ode_solver.is_direct_collocation and interpolation_type != InterpolationType.EACH_FRAME:
            #         all_nx = nx * nlp.ns * (nlp.ode_solver.polynomial_degree + 1) + nx
            #         outer_offset = nx * (nlp.ode_solver.polynomial_degree + 1)
            #         repeat = nlp.ode_solver.polynomial_degree + 1
            #     else:
            #         all_nx = nx * (nlp.ns + 1)
            #         outer_offset = nx
            #         repeat = 1
            #
            #     x_init = InitialGuess([0] * all_nx, interpolation=InterpolationType.CONSTANT)
            #     for k in range(nlp.ns + 1):
            #         for p in range(repeat if k != nlp.ns else 1):
            #             span = slice(k * outer_offset + p * nx, k * outer_offset + (p + 1) * nx)
            #             point = k if k != 0 else 0 if p == 0 else 1
            #             if isinstance(nlp.x_init, NoisedInitialGuess):
            #                 if nlp.x_init.type == InterpolationType.ALL_POINTS:
            #                     point = k * repeat + p
            #             elif isinstance(nlp.x_init, InitialGuess) and nlp.x_init.type == InterpolationType.EACH_FRAME:
            #                 point = k * repeat + p
            #             x_init.init[span, 0] = nlp.x_init.init.evaluate_at(shooting_point=point)
            #     self.x_init[i_phase] = x_init
            #
            #     #probleme ici sur self.x_init
            #     # il y a nlp.ns+1 elements pour chaque phase, il faut pareil partout

            # For controls
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type == ControlType.CONSTANT:
                    ns = nlp.ns
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    ns = nlp.ns + 1
                else:
                    raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
                nu = nlp.controls.shape
                all_nu = nu * ns
                u_init = InitialGuess([0] * all_nu, interpolation=InterpolationType.CONSTANT)
                for k in range(ns):
                    u_init.init[k * nu : (k + 1) * nu, 0] = nlp.u_init.init.evaluate_at(shooting_point=k)

                self.u_init[i_phase] = u_init

            # Code from L.Secohoire
            # else :
            #     if nlp.control_type == ControlType.CONSTANT:
            #         ns = nlp.ns
            #     elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
            #         ns = nlp.ns + 1
            #     else:
            #         raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
            #     #nu = nlp.controls.shape
            #
            #     nu = len(index_roots_u)
            #     all_nu = nu * ns
            #     u_init = InitialGuess([0] * all_nu, interpolation=InterpolationType.CONSTANT)
            #     for k in range(ns):
            #         u_init.init[k * nu : (k + 1) * nu, 0] = nlp.u_init.init.evaluate_at(shooting_point=k)
            #
            #     self.u_init[i_phase] = u_init

    def add_parameter(self, param: Parameter):
        """
        Add a parameter to the parameters pool

        Parameters
        ----------
        param: Parameter
            The new parameter to add to the pool
        """

        ocp = self.ocp
        param.cx = param.cx if param.cx is not None else ocp.cx.sym(param.name, param.size, 1)
        param.mx = MX.sym(f"{param.name}_MX", param.size, 1)

        if param.name in self.parameters_in_list:
            # Sanity check, you can only add a parameter with the same name if they do the same thing
            i = self.parameters_in_list.index(param.name)

            if param.function != self.parameters_in_list[i].function:
                raise RuntimeError("Pre dynamic function of same parameters must be the same")
            self.parameters_in_list[i].size += param.size
            self.parameters_in_list[i].cx = vertcat(self.parameters_in_list[i].cx, param.cx)
            self.parameters_in_list[i].mx = vertcat(self.parameters_in_list[i].mx, param.mx)
            self.parameters_in_list[i].scaling = vertcat(self.parameters_in_list[i].scaling, param.scaling)
            if param.params != self.parameters_in_list[i].params:
                raise RuntimeError("Extra parameters of same parameters must be the same")
            self.parameters_in_list[i].bounds.concatenate(param.bounds)
            self.parameters_in_list[i].initial_guess.concatenate(param.initial_guess)
        else:
            self.parameters_in_list.add(param)
