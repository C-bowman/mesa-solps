from mesa.objectives import ObjectiveFunction
from sims.interface import SolpsInterface
from sims.instruments import Instrument
from sims.likelihoods import cauchy_likelihood


class SolpsLikelihood(ObjectiveFunction):
    def __init__(self, diagnostics: list[Instrument]):
        self.diagnostics = diagnostics
        self.name = "cauchy_logprob"

    def evaluate(self, simulation_results: SolpsInterface) -> dict[str, float]:
        # update the diagnostics with the latest SOLPS data
        for dia in self.diagnostics:
            dia.update_interface(simulation_results)

        # calculate the log-probability
        cauchy_logprob = sum([dia.log_likelihood(likelihood=cauchy_likelihood) for dia in self.diagnostics])
        return {
            "cauchy_logprob": cauchy_logprob,
        }