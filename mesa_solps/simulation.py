from pathlib import Path
import os
from os.path import isfile
from time import time
import logging
import subprocess
import shutil as sh

from sims.interface import SolpsInterface
from mesa.simulations import RunStatus, Simulation, SimulationRun
from .transport import write_solps_transport_inputfile
from .models import linear_transport_profile, profile_radius_axis
from .parameters import conductivity_profile, diffusivity_profile, required_parameters


class SolpsRun(SimulationRun):
    def __init__(
        self,
        parameters: dict[str, float],
        run_number: int,
        directory: Path,
        launch_time: float,
        run_id: str,
        timeout_hours: int
    ):
        self.parameters = parameters
        self.run_number = run_number
        self.directory = directory
        self.launch_time = launch_time
        self.run_id = run_id
        self.timeout_hours = timeout_hours

    def status(self) -> RunStatus:
        whoami = subprocess.run("whoami", capture_output=True, encoding="utf-8")
        username = whoami.stdout.rstrip()

        squeue = subprocess.run(
            ["squeue", "-u", username], capture_output=True, encoding="utf-8"
        )
        job_queue = squeue.stdout

        if self.run_id not in job_queue:
            balance_created = isfile(self.directory / "balance.nc")
            status = "complete" if balance_created else "crashed"
        elif (time() - self.launch_time) > self.timeout_hours * 3600.:
            status = "timed-out"
        else:
            status = "running"
        return status

    def cleanup(self):
        output_files = [f for f in os.listdir(self.directory) if isfile(self.directory / f)]
        allowed_files = (
            "balance.nc", "input.dat", "b2.neutrals.parameters",
            "b2.boundary.parameters", "b2.numerics.parameters",
            "b2.transport.parameters", "b2.transport.inputfile",
            "b2mn.dat"
        )
        deletions = [f for f in output_files if f not in allowed_files]
        [os.remove(self.directory / f) for f in deletions]

    def get_results(self) -> SolpsInterface:
        results_path = self.directory / "balance.nc"
        return SolpsInterface(balance_filepath=results_path)

    def cancel(self):
        subprocess.run(["scancel", self.run_id])

    def __key(self):
        return self.run_id, self.directory, self.run_number, self.launch_time

    def __hash__(self):
        return hash(self.__key())


class Solps(Simulation):
    def __init__(
        self,
        reference_directory: Path,
        transport_profile_bounds: tuple[float, float],
        set_div_transport: bool = False,
        n_proc: int = 1,
        memory_gb: int = 20,
        timeout_hours: int = 24,
    ):
        self.reference_directory = reference_directory
        self.transport_bounds = transport_profile_bounds
        self.set_div_transport = set_div_transport
        self.n_proc = n_proc
        self.memory = memory_gb
        self.timeout_hours = timeout_hours

    def launch(
        self,
        run_number: int,
        simulations_directory: Path,
        parameters: dict,
    ) -> SolpsRun:
        """
        Evaluates SOLPS for the provided transport profiles and saves the results.

        :param int run_number: \
            The run number corresponding to the requested SOLPS run, used to name
            directory in which the SOLPS output is stored.

        :param str simulations_directory: \

        """
        case_dir = simulations_directory / f"run_{run_number}"

        build_solps_case(
            reference_directory=self.reference_directory,
            case_directory=case_dir,
            parameter_dictionary=parameters
        )

        # produce transport profiles defined by new point
        chi_params = [parameters[k] for k in conductivity_profile]
        chi_radius = profile_radius_axis(chi_params, self.transport_bounds)
        chi = linear_transport_profile(chi_radius, chi_params, self.transport_bounds)

        D_params = [parameters[k] for k in diffusivity_profile]
        D_radius = profile_radius_axis(D_params, self.transport_bounds)
        D = linear_transport_profile(D_radius, D_params, self.transport_bounds)

        write_solps_transport_inputfile(
            filename=case_dir / 'b2.transport.inputfile',
            grid_dperp=D_radius,
            values_dperp=D,
            grid_chieperp=chi_radius,
            values_chieperp=chi,
            grid_chiiperp=chi_radius,
            values_chiiperp=chi,
            set_ana_visc_dperp=False,  # TODO - may need to be chosen via settings file
            no_pflux=True,
            no_div=self.set_div_transport
        )

        # Go to the SOLPS run directory to prepare execution
        os.chdir(case_dir)

        command = "itmsubmit" if self.n_proc == 1 else f'itmsubmit -m "-np {self.n_proc}" -M {self.memory}GB'
        start_run = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

        start_run_output = start_run.communicate()[0]
        os.chdir(simulations_directory)
        # Find the batch job number
        findstr = 'Submitted batch job'
        tmp = str(start_run_output).find(findstr)
        job_id = str(start_run_output)[tmp + len(findstr) + 1:-3]

        logging.info(f'[solps_interface] Submitted job {job_id}')
        return SolpsRun(
            run_id=job_id,
            directory=case_dir,
            run_number=run_number,
            parameters=parameters,
            launch_time=time(),
            timeout_hours=self.timeout_hours
        )


def build_solps_case(
    reference_directory: Path,
    case_directory: Path,
    parameter_dictionary: dict
):
    input_files = [
        "input.dat",
        "fort.1",
        "fort.13",
        "b2.neutrals.parameters",
        "b2.boundary.parameters",
        "b2.numerics.parameters",
        "b2.transport.parameters",
        "b2mn.dat"
    ]

    # create the case directory and copy all the reference files
    case_directory.mkdir()
    sh.copy(reference_directory / "b2fstate", case_directory / "b2fstati")
    for input in input_files:
        if isfile(reference_directory / input):
            sh.copy(reference_directory / input, case_directory / input)

    optional_params = {k for k in parameter_dictionary.keys()} - required_parameters
    written_params = set()

    mesa_input_files = [f + ".mesa" for f in input_files]
    mesa_input_files = [f for f in mesa_input_files if isfile(reference_directory / f)]

    if len(optional_params) != 0:
        if len(mesa_input_files) == 0:
            raise FileNotFoundError(
                f"""\n
                \r[ MESA ERROR ]
                \r>> The following optional parameters were specified
                \r>> {optional_params}
                \r>> However no input files with a .mesa extension were
                \r>> found in the reference directory.
                """
            )

        for mif in mesa_input_files:
            output = []
            with open(reference_directory / mif) as f:
                for line in f:
                    for p in optional_params:
                        if '{'+p+'}' in line:
                            val_string = parameter_dictionary[p]  # TODO - CONVERT TO STRING WITH FORMATTING
                            line.replace('{'+p+'}', val_string)
                            written_params.add(p)
                    output.append(line)

            with open(case_directory / mif, 'w') as f:  # TODO - does this overwrite?
                for item in output:
                    f.write("%s\n" % item)

        unwritten_params = optional_params - written_params
        if len(unwritten_params) > 0:
            raise ValueError(
                f"""\n
                \r[ MESA ERROR ]
                \r>> The following optional parameters were specified
                \r>> {unwritten_params}
                \r>> but were not found in any input files with a .mesa extension.
                """
            )

