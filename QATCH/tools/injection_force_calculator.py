from QATCH.tools.web_viewer import BaseAcceleratedCalculator


class InjectionForceCalculatorModule(BaseAcceleratedCalculator):
    def __init__(self):
        super().__init__(
            title="Injection Force Calculator",
            target_url="https://qatch-technologies.github.io/Injection-Force-Calculator/injection_force_calculator.html",
        )
