import os
import numpy as np
from bioptim import (
    Solver,
)


def test_custom_model():
    from bioptim.examples.custom_model import main as ocp_module
    from bioptim.examples.custom_model.custom_package import my_model as model
    from bioptim.examples.custom_model.custom_package import (
        custom_dynamics as dynamics,
        custom_configure_my_dynamics as configure_dynamics,
    )

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        model=model.MyModel(),
        final_time=1,
        n_shooting=30,
        configure_dynamics=configure_dynamics,
        dynamics=dynamics,
    )

    np.testing.assert_almost_equal(ocp.nlp[0].model.nb_q, 1)
    np.testing.assert_almost_equal(ocp.nlp[0].model.nb_qdot, 1)
    np.testing.assert_almost_equal(ocp.nlp[0].model.nb_qddot, 1)
    np.testing.assert_almost_equal(ocp.nlp[0].model.nb_tau, 1)
    assert ocp.nlp[0].model.nb_quaternions == 0  # added by the ocp because it must be in any BioModel
    np.testing.assert_almost_equal(ocp.nlp[0].model.mass, 1)
    assert ocp.nlp[0].model.name_dof == ["rotx"]

    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(2)
    sol = ocp.solve(solver=solver)

    np.testing.assert_almost_equal(sol.states["q"][0, 0], np.array([0]))
    np.testing.assert_almost_equal(sol.states["q"][0, -1], np.array([3.14]))
    np.testing.assert_almost_equal(sol.states["qdot"][0, 0], np.array([0]))
    np.testing.assert_almost_equal(sol.states["qdot"][0, -1], np.array([0]))
