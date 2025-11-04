"""
optimizer.py

Module for optimizing bioformulation parameters to match a target viscosity profile.

This module defines the `Optimizer` class, which uses differential evolution to
search the formulation feature space under specified constraints, minimizing
the mean squared error between predicted and target viscosity profiles.

Enhanced with real-time progress visualization support.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
    Enhanced by: Claude

Date:
    2025-11-04

Version:
    1.2
"""
import multiprocessing
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from scipy.optimize import differential_evolution

try:
    from src.models.formulation import ViscosityProfile, Formulation
    from src.models.predictor import Predictor
    from src.models.ingredient import Ingredient
    from src.utils.constraints import Constraints
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile, Formulation
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.models.ingredient import Ingredient
    from QATCH.VisQAI.src.utils.constraints import Constraints


class OptimizationStatus:
    """Data class to hold optimization status information."""

    def __init__(self, iteration: int, num_iterations: int, best_value: float,
                 population_size: int, convergence: float):
        self.iteration = iteration
        self.num_iterations = num_iterations
        self.best_value = best_value
        self.population_size = population_size
        self.convergence = convergence
        self.progress_percent = (
            iteration / num_iterations * 100) if num_iterations > 0 else 0

    def __repr__(self) -> str:
        return (f"OptimizationStatus(iteration={self.iteration}, "
                f"best_value={self.best_value:.6f}, "
                f"progress={self.progress_percent:.1f}%)")


class OptimizationProgressTracker:
    """Track and manage optimization progress with history."""

    def __init__(self):
        self.history = []
        self.start_value = None
        self.best_value = None
        self.iterations = []
        self.best_values = []

    def __call__(self, status: OptimizationStatus):
        """Track optimization progress."""
        self.history.append(status)
        self.iterations.append(status.iteration)
        self.best_values.append(status.best_value)

        if self.start_value is None:
            self.start_value = status.best_value

        if self.best_value is None or status.best_value < self.best_value:
            self.best_value = status.best_value

        # Print progress
        improvement = self.start_value - status.best_value if self.start_value else 0
        print(f"[{status.iteration:3d}/{status.num_iterations}] "
              f"Best: {status.best_value:10.6f} | "
              f"Improvement: {improvement:10.6f} | "
              f"Progress: {status.progress_percent:6.1f}%")

    def get_improvement_rate(self) -> float:
        """Calculate average improvement per iteration."""
        if len(self.history) < 2 or self.start_value is None:
            return 0.0
        return (self.start_value - self.best_value) / len(self.history)

    def get_stagnation_iterations(self, threshold: int = 10) -> int:
        """Count iterations without improvement in last 'threshold' iterations."""
        if len(self.history) < threshold:
            return 0

        recent = self.history[-threshold:]
        best_recent = min(s.best_value for s in recent)

        stagnant_count = 0
        for status in reversed(recent):
            if status.best_value == best_recent:
                stagnant_count += 1
            else:
                break

        return stagnant_count

    def get_plot_data(self) -> tuple:
        """Get data for plotting optimization progress.

        Returns:
            tuple: (iterations, best_values) for plotting
        """
        return (self.iterations.copy(), self.best_values.copy())


class Optimizer:
    """
    Performs constrained optimization of formulation features to match a desired viscosity profile.

    Attributes:
        constraints (Constraints): Defines allowable feature ranges and categorical choices.
        predictor (Predictor): Model predictor for generating viscosity predictions.
        target (ViscosityProfile): Desired viscosity profile to approximate.
        maxiter (int): Maximum iterations for the differential evolution solver.
        popsize (int): Population size multiplier for the solver.
        tol (float): Convergence tolerance threshold.
        seed (int): Random seed for reproducibility.
        bounds (List[Tuple[float, float]]): Numeric bounds extracted from constraints.
        encoding (List[Dict]): Encoding metadata for each feature (type, feature name, choices).
        cat_choices (Dict[str, List[Ingredient]]): Categorical choices for each categorical feature.
    """

    def __init__(
        self,
        constraints: Constraints,
        predictor: Predictor,
        target: ViscosityProfile,
        maxiter: int = 100,
        popsize: int = 15,
        tol: float = 1e-6,
        seed: int = None,
    ):
        """
        Initializes the Optimizer with constraints, predictor, and target profile.

        Args:
            constraints (Constraints): Constraint definitions for sampling features.
            predictor (Predictor): Predictor model for inferring viscosity profiles.
            target (ViscosityProfile): Target viscosity profile for optimization.
            maxiter (int, optional): Maximum number of solver iterations. Defaults to 100.
            popsize (int, optional): Differential evolution population multiplier. Defaults to 15.
            tol (float, optional): Convergence tolerance. Defaults to 1e-6.
            seed (int, optional): Random seed. Defaults to None.
        """
        self.constraints = constraints
        self.predictor = predictor
        self.target = target
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.seed = seed

        # Build numeric bounds and encoding metadata
        self.bounds, self.encoding = self.constraints.build()

        # Prepare categorical choices for each categorical feature
        all_ings = self.constraints._ingredient_ctrl.get_all_ingredients()
        self.cat_choices: Dict[str, List[Ingredient]] = {}
        for enc in self.encoding:
            if enc["type"] == "cat":
                feat = enc["feature"]
                choices = self.constraints._choices.get(feat)
                if choices is None:
                    cls = self.constraints._FEATURE_CLASS[feat]
                    choices = [ing for ing in all_ings if isinstance(ing, cls)]
                self.cat_choices[feat] = choices

    def _decode(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Decodes a numeric vector into a feature-to-value mapping.

        Args:
            x (np.ndarray): Array of optimization variables.

        Returns:
            Dict[str, Any]: Mapping from feature names to decoded values (floats or Ingredients).
        """
        out: Dict[str, Any] = {}
        for xi, enc in zip(x, self.encoding):
            feat = enc["feature"]
            if enc["type"] == "cat":
                # Round and clip to valid index range for categorical choices
                idx = int(np.clip(round(xi), 0, len(
                    self.cat_choices[feat]) - 1))
                out[feat] = self.cat_choices[feat][idx]
            else:
                out[feat] = float(xi)
        return out

    def _build_formulation(self, feat_dict: Dict[str, Any]) -> Formulation:
        """
        Constructs a Formulation object from decoded feature values.

        Args:
            feat_dict (Dict[str, Any]): Mapping of feature names to decoded values.

        Returns:
            Formulation: Formulation object with components set according to feat_dict.
        """
        form = Formulation()
        form.set_buffer(
            buffer=feat_dict["Buffer_type"],
            concentration=feat_dict["Buffer_conc"],
            units="mM"
        )
        form.set_protein(
            protein=feat_dict["Protein_type"],
            concentration=feat_dict["Protein_conc"],
            units="mg/mL"
        )
        form.set_stabilizer(
            stabilizer=feat_dict["Stabilizer_type"],
            concentration=feat_dict["Stabilizer_conc"],
            units="M"
        )
        form.set_salt(
            salt=feat_dict["Salt_type"],
            concentration=feat_dict["Salt_conc"],
            units="mM"
        )
        form.set_surfactant(
            surfactant=feat_dict["Surfactant_type"],
            concentration=feat_dict["Surfactant_conc"],
            units="%w"
        )
        form.set_excipient(
            excipient=feat_dict["Excipient_type"],
            concentration=feat_dict["Excipient_conc"],
            units="mM"
        )
        form.set_temperature(
            temp=feat_dict["Temperature"]
        )
        return form

    def _mse_loss(self, prof1: ViscosityProfile, prof2: ViscosityProfile) -> float:
        """
        Computes the mean squared error between two viscosity profiles.

        Args:
            prof1 (ViscosityProfile): Profile to interpolate from.
            prof2 (ViscosityProfile): Profile to compare to.

        Returns:
            float: Mean squared difference of viscosities at prof2's shear rates.
        """
        v1 = np.interp(prof2.shear_rates, prof1.shear_rates, prof1.viscosities)
        return float(np.mean((v1 - prof2.viscosities) ** 2))

    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function for the optimizer: MSE between predicted and target profiles.

        Args:
            x (np.ndarray): Candidate variable vector.

        Returns:
            float: Loss value (MSE) for this candidate.
        """
        feat_dict = self._decode(x)
        formulation = self._build_formulation(feat_dict)
        pred = self.predictor.predict(
            df=formulation.to_dataframe(training=False))
        pred_vp = ViscosityProfile(
            shear_rates=[100, 1000, 10000, 100000, 15000000],
            viscosities=pred.flatten().tolist()
        )
        return self._mse_loss(pred_vp, self.target)

    def optimize(self, strategy: str = 'best1bin', mutation: tuple = (0.5, 1),
                 recombination: float = 0.7, init: str = 'latinhypercube',
                 atol: int = 0, workers: int = 1,
                 progress_callback: Optional[Callable[[OptimizationStatus], None]] = None) -> Formulation:
        """
        Executes the differential evolution solver to find the best formulation.

        Args:
            strategy (str): DE strategy to use. Defaults to 'best1bin'.
            mutation (tuple): Mutation rate range. Defaults to (0.5, 1).
            recombination (float): Recombination rate. Defaults to 0.7.
            init (str): Initialization method. Defaults to 'latinhypercube'.
            atol (int): Absolute tolerance. Defaults to 0.
            workers (int): Number of workers. Defaults to 1.
            progress_callback (Optional[Callable]): Callback function to report progress.
                                                     Receives OptimizationStatus objects.

        Returns:
            Formulation: Best formulation found by optimization.
        """
        # Create progress tracker
        tracker = OptimizationProgressTracker()

        # Create callback wrapper for differential_evolution
        def callback_wrapper(xk, convergence=0):
            # Calculate best value from current solution
            best_value = self._objective(xk)

            # Create status with actual objective value
            status = OptimizationStatus(
                iteration=len(tracker.history) + 1,
                num_iterations=self.maxiter,
                best_value=best_value,
                population_size=self.popsize,
                convergence=convergence
            )

            # Track progress
            tracker(status)

            # Call user's progress callback if provided
            if progress_callback:
                progress_callback(status)
        result = differential_evolution(
            func=self._objective,
            bounds=self.bounds,
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=self.tol,
            seed=self.seed,
            polish=True,
            disp=False,
            workers=int(workers),
            atol=atol,
            recombination=recombination,
            mutation=mutation,
            strategy=strategy,
            init=init,
            callback=callback_wrapper,
        )

        best_feats = self._decode(result.x)
        return self._build_formulation(best_feats)
