from typing import Any
from abc import ABC, abstractmethod

from casadi import MX

from ...misc.options import UniquePerPhaseOptionList, OptionDict, OptionGeneric
from ...misc.enums import VariableType


class FatigueModel(ABC):
    def __init__(self, scaling: float = 1, state_only: bool = None, apply_to_joint_dynamics: bool = None):
        """
        scaling: float
            A scaling factor to apply
        """
        self.scaling = scaling
        self.state_only = self.default_state_only() if state_only is None else state_only
        self.apply_to_joint_dynamics = (
            self.default_apply_to_joint_dynamics() if apply_to_joint_dynamics is None else apply_to_joint_dynamics
        )

    @staticmethod
    @abstractmethod
    def type() -> str:
        """
        The type of Fatigue
        """

    @staticmethod
    @abstractmethod
    def suffix(variable_type: VariableType) -> tuple:
        """
        The type of Fatigue
        """

    @staticmethod
    @abstractmethod
    def color() -> tuple:
        """
        The coloring when drawn
        """

    @abstractmethod
    def default_state_only(self) -> bool:
        """
        What is the default value for state_only
        """

    @abstractmethod
    def default_apply_to_joint_dynamics(self) -> bool:
        """
        What is the default value for apply_to_joint_dynamics
        """

    @abstractmethod
    def default_initial_guess(self) -> tuple:
        """
        The initial guess the fatigue parameters are expected to have
        """

    @abstractmethod
    def default_bounds(self, variable_type: VariableType) -> tuple:
        """
        The bounds the fatigue parameters are expected to have
        """

    @staticmethod
    @abstractmethod
    def dynamics_suffix() -> str:
        """
        The suffix that is appended to the variable name that describes the dynamics
        """

    @staticmethod
    @abstractmethod
    def fatigue_suffix() -> str:
        """
        The suffix that is appended to the variable name that describes the fatigue
        """

    @abstractmethod
    def dynamics(self, dxdt, nlp, index, states, controls) -> MX:
        """
        Augment the dxdt vector with the derivative of the fatigue states

        Parameters
        ----------
        dxdt: MX
            The MX vector to augment
        nlp: NonLinearProgram
            The current phase
        index: int
            The index of the current fatigue element
        states: OptionVariable
            The state variable
        controls: OptionVariable
            The control variable

        Returns
        -------
        dxdt augmented
        """

    @property
    @abstractmethod
    def multi_type(self):
        """
        The associate MultiFatigueModel to the model

        Returns
        -------
        The associate MultiFatigueModel to the model
        """


class MultiFatigueModel(OptionGeneric):
    def __init__(
        self,
        model: FatigueModel | list,
        state_only: bool,
        split_controls: bool = True,
        apply_to_joint_dynamics: bool = False,
        **params
    ):
        """
        model: FatigueModel
            The actual fatigue model
        state_only: bool
            If the added fatigue should be used in the dynamics or only computed
        apply_to_joint_dynamics: bool
            If the fatigue should be applied to the joint (only makes sense for muscle fatigue)
        suffix_default: str
            The replacement of suffix if any, for internal purpose
        split_controls: bool
            If the tau should be separated into minus and plus part or use an if_else case
        apply_to_joint_dynamics: bool
            If the fatigue should be applied to the dynamics of the system
        params: Any
            Any other parameters to pass to OptionGeneric
        """

        super(MultiFatigueModel, self).__init__(**params)
        if isinstance(model, FatigueModel):
            model = [model]

        if self.suffix():
            # If there is suffix, convert to dictionary
            model_tp = {}
            for i, key in enumerate(self.suffix()):
                if key:
                    model_tp[key] = model[i]
                else:
                    model_tp[key] = model
        else:
            model_tp = model

        self.models = model_tp
        self.state_only = state_only
        self.apply_to_joint_dynamics = apply_to_joint_dynamics
        self.split_controls = split_controls

    @property
    def shape(self):
        return len(self.models)

    @staticmethod
    @abstractmethod
    def model_type() -> str:
        """
        The type of Fatigue
        """

    @staticmethod
    @abstractmethod
    def color() -> tuple:
        """
        The color to be draw
        """

    @staticmethod
    @abstractmethod
    def plot_factor() -> tuple:
        """
        The factor to multiply the plots so it is not one over another
        """

    @abstractmethod
    def suffix(self) -> tuple:
        """
        The type of Fatigue
        """

    def add(self, fatigue: FatigueModel):
        """
        Add a new element to the fatigue list

        Parameters
        ----------
        fatigue: FatigueModel
            The model to add
        """
        self.models.append(fatigue)

    def dynamics(self, dxdt, nlp, index, states, controls):
        for suffix in self.suffix():
            dxdt = self._dynamics_per_suffix(dxdt, suffix, nlp, index, states, controls)

        return dxdt

    @abstractmethod
    def _dynamics_per_suffix(self, dxdt, suffix, nlp, index, states, controls):
        """

        Parameters
        ----------
        dxdt: MX
            The MX vector to augment
        suffix: str
            The str for each suffix
        nlp: NonLinearProgram
            The current phase
        index: int
            The index of the current fatigue element
        states: OptionVariable
            The state variable
        controls: OptionVariable
            The control variable

        Returns
        -------

        """

    @staticmethod
    @abstractmethod
    def default_state_only():
        """
        What is the default value for state_only
        """

    @staticmethod
    @abstractmethod
    def default_apply_to_joint_dynamics():
        """
        What is the default value for apply_to_joint_dynamics
        """

    @abstractmethod
    def default_bounds(self, index: int, variable_type: VariableType) -> tuple:
        """
        The default bounds for the index element in models

        Parameters
        ----------
        index: int
            The index of the element
        variable_type: VariableType
            The type of variable

        Returns
        -------
        The default bounds
        """

    @abstractmethod
    def default_initial_guess(self, index: int, variable_type: VariableType):
        """
        The default initial guess for the index element in models

        Parameters
        ----------
        index: int
            The index of the element
        variable_type: VariableType
            The type of variable

        Returns
        -------
        The default initial guess
        """

    def _convert_to_models_key(self, item: int | str):
        """
        Convert the item to a key if self.models is a dictionary, based on suffix() order

        Parameters
        ----------
        item: int | str
            The item to convert

        Returns
        -------
        The usable key
        """
        if isinstance(self.models, dict):
            return list(self.models.keys())[item]
        else:
            return item


class MultiFatigueInterface(MultiFatigueModel, ABC):
    def suffix(self) -> tuple:
        return ("fatigue",)

    def default_bounds(self, index: int, variable_type: VariableType) -> tuple:
        return self.models["fatigue"].default_bounds(variable_type)

    def default_initial_guess(self, index: int, variable_type: VariableType):
        return self.models["fatigue"].default_initial_guess()

    def _dynamics_per_suffix(self, dxdt, suffix, nlp, index, states, controls):
        return self.models["fatigue"].dynamics(dxdt, nlp, index, states, controls)

    @staticmethod
    def color() -> tuple:
        return ("tab:orange",)

    @staticmethod
    def plot_factor() -> tuple:
        return (1,)


class FatigueUniqueList(UniquePerPhaseOptionList):
    def print(self):
        NotImplementedError("FatigueUniqueList cannot be printed")

    def __init__(self, suffix: list | tuple):
        """
        Parameters
        ----------
        suffix: list
            The type of Fatigue
        """

        super(FatigueUniqueList, self).__init__()
        self.suffix = suffix

    def add(self, **extra_arguments: Any):
        self._add(option_type=MultiFatigueModel, state_only=None, apply_to_joint_dynamics=None, **extra_arguments)

    def __next__(self) -> Any:
        """
        Get the next option of the list

        Returns
        -------
        The next option of the list
        """
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1][0] if self.options[self._iter_idx - 1] else None

    def dynamics(self, dxdt, nlp, states, controls):
        for i, elt in enumerate(self):
            dxdt = elt.models.dynamics(dxdt, nlp, i, states, controls)
        return dxdt


class FatigueList(OptionDict):
    def add(
        self,
        model: FatigueModel | MultiFatigueModel,
        index: int = -1,
        state_only: bool = None,
        apply_to_joint_dynamics: bool = None,
    ):
        """
        Add a muscle to the dynamic list

        Parameters
        ----------
        model: FatigueModel | MultiFatigueModel
            The dynamics to add, if more than one dynamics is required, a list can be sent
        index: int
            The index of the muscle, referring to the muscles order in the bioMod
        state_only: bool
            If the added fatigue should be used in the dynamics or only computed
        apply_to_joint_dynamics: bool
            If the fatigue should be applied to joint torque directly
        """

        if state_only is None:
            state_only = model.default_state_only()

        if apply_to_joint_dynamics is None:
            apply_to_joint_dynamics = model.default_apply_to_joint_dynamics()

        if isinstance(model, FatigueModel):
            model = model.multi_type(model, state_only=state_only, apply_to_joint_dynamics=apply_to_joint_dynamics)

        if model.model_type() not in self.options[0]:
            self.options[0][model.model_type()] = FatigueUniqueList(model.suffix())

        self.options[0][model.model_type()].add(model=model, phase=index)

    def dynamics(self, dxdt, nlp, index, states, controls):
        raise NotImplementedError("FatigueDynamics is abstract")

    def __contains__(self, item):
        return item in self.options[0]

    def __getitem__(self, item: int | str | list | tuple) -> FatigueUniqueList:
        return super(FatigueList, self).__getitem__(item)

    def print(self):
        raise NotImplementedError("Printing is not implemented yet for FatigueList")
