from mesa.core import Mesa

# ----------------------------------------------------------------------------
#   path settings
# ----------------------------------------------------------------------------
from pathlib import Path
# directory where a reference SOLPS run is stored
simulations_directory = Path('/pfs/work/g2dmoult/solps-iter_3.0.9_develop/runs/mastu_sxd_46860_450ms_Donly')

# file name in which the training data will be stored
evaluations_filepath = simulations_directory / "evaluations_data.h5"


# ----------------------------------------------------------------------------
#   diagnostics settings
# ----------------------------------------------------------------------------
from numpy import load
from sims.instruments import ThomsonScattering
from mesa_solps.simulation import Solps
from mesa_solps.objective import SolpsLikelihood

instrument_data = load('/pfs/work/g2dmoult/mesa-solps/testing_ts_instrument_data.npz')
TS = ThomsonScattering(
    R=instrument_data['R'],
    z=instrument_data['z'],
    weights=instrument_data['weights'],
    measurements=load('/pfs/work/g2dmoult/mesa-solps/testing_ts_data.npz')
)

objective_function = SolpsLikelihood(diagnostics=[TS])


# ----------------------------------------------------------------------------
#   simulation settings
# ----------------------------------------------------------------------------
simulation = Solps(
    n_proc=36,
    memory_gb=20,
    timeout_hours=24,
    set_div_transport=True,
    transport_profile_bounds=(-0.250, 0.240),
    reference_directory=simulations_directory / "ref_clean"
)


# ----------------------------------------------------------------------------
#   strategy settings
# ----------------------------------------------------------------------------
from mesa.strategies import GPOptimizer
from inference.gp import SquaredExponential, QuadraticMean, UpperConfidenceBound

strategy = GPOptimizer(
    covariance_kernel=SquaredExponential(),
    mean_function=QuadraticMean(),
    acquisition_function=UpperConfidenceBound(kappa=1.),
    initial_sample_count=20,
    cross_validation=False,
    trust_region_width=0.3,
    n_processes=1,
)


# ----------------------------------------------------------------------------
#   parameter settings
# ----------------------------------------------------------------------------
# parameters to fix or vary. To fix set as a single value. To vary set bounds as a tuple
parameters = {
    # Chi-profile parameter boundaries
    'chi_boundary_left'  : (0., 6.),       # left boundary height from barrier level
    'chi_boundary_right' : (0., 15.),      # right boundary height from barrier level
    'chi_frac_left'      : (0.25, 1.),     # left-middle height as a fraction of barrier-boundary gap
    'chi_frac_right'     : (0.25, 1.),     # right-middle height as a fraction of barrier-boundary gap
    'chi_barrier_centre' : (-0.05, 0.05),  # transport barrier centre
    'chi_barrier_height' : (1e-3, 0.2),    # transport barrier height
    'chi_barrier_width'  : (0.002, 0.04),  # transport barrier width
    'chi_gap_left'       : (2e-3, 0.05),   # radius gap between left-midpoint and transport barrier
    'chi_gap_right'      : (2e-3, 0.05),   # radius gap between right-midpoint and transport barrier

    # D-profile parameter boundaries
    'D_boundary_left'  : (0., 4.),       # left boundary height from barrier level
    'D_boundary_right' : (0., 2.),       # right boundary height from barrier level
    'D_frac_left'      : (0.25, 1.),     # left-middle height as a fraction of barrier-boundary gap
    'D_frac_right'     : (0.25, 1.),     # right-middle height as a fraction of barrier-boundary gap
    'D_barrier_centre' : (-0.05, 0.05),  # transport barrier centre
    'D_barrier_height' : (1e-3, 0.2),    # transport barrier height
    'D_barrier_width'  : (0.002, 0.04),  # transport barrier width
    'D_gap_left'       : (2e-3, 0.05),    # radius gap between left-midpoint and transport barrier
    'D_gap_right'      : (2e-3, 0.05),    # radius gap between right-midpoint and transport barrier
}

mesa = Mesa(
    parameters=parameters,
    simulation=simulation,
    objective_function=objective_function,
    strategy=strategy,
    simulations_directory=simulations_directory,
    evaluations_filepath=evaluations_filepath,
    max_concurrent_runs=4,
    max_iterations=60
)

mesa.run()
