from QATCH.tools.web_viewer import BaseAcceleratedCalculator


class DonnanCalculatorModule(BaseAcceleratedCalculator):
    def __init__(self):
        super().__init__(
            title="Donnan Calculator",
            target_url="https://qatch-technologies.github.io/Donnan-Calculator/donnan_calculator.html",
        )
