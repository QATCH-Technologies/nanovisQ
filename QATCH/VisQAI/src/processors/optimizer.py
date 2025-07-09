"""
optimizer.py

Module for optimizing bioformulation parameters to match a target viscosity profile.

This module defines the `Optimizer` class, which uses differential evolution to
search the formulation feature space under specified constraints, minimizing
the mean squared error between predicted and target viscosity profiles.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-07-09

Version:
    1.0
"""

from typing import Dict, Any, List
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
        pred = self.predictor.predict(data=formulation.to_dataframe())
        pred_vp = ViscosityProfile(
            shear_rates=[100, 1000, 10000, 100000, 15000000],
            viscosities=pred.flatten().tolist()
        )
        return self._mse_loss(pred_vp, self.target)

    def optimize(self) -> Formulation:
        """
        Executes the differential evolution solver to find the best formulation.

        Returns:
            Formulation: Best formulation found by optimization.
        """
        result = differential_evolution(
            func=self._objective,
            bounds=self.bounds,
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=self.tol,
            seed=self.seed,
            polish=True,
        )
        best_feats = self._decode(result.x)
        return self._build_formulation(best_feats)
