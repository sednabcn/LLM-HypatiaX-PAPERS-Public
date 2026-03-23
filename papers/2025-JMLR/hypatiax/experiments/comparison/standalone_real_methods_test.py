#!/usr/bin/env python3
"""
STANDALONE TEST SUITE - IMPORTS REAL METHODS DIRECTLY (v4 - EXTRAPOLATION FIXED!)
===================================================================================

FULL FIX for extrapolation prediction issues!

Changes in v4:
- CRITICAL FIX: Store prediction data in TestResult._prediction_cache instead of instance variables
- This prevents cache conflicts when the same wrapper runs multiple tests
- Fixed PureLLMMethodWrapper._predict() with robust error handling
- Fixed HybridSystemV40Wrapper._predict() with proper variable name resolution
- Fixed NeuralNetworkMethodWrapper._predict() with shape validation
- Enhanced _evaluate_extrapolation() with verbose debugging
- Better filtering of inf values in final summary
- Added diagnostic output to track prediction failures

The Root Problem (SOLVED):
- Method wrappers are singleton instances shared across all tests
- When test 2 runs, it overwrites the cached data from test 1
- Extrapolation tries to use cached data from test 1, but it's gone!
- Solution: Store prediction artifacts in each TestResult object

Usage:
    python standalone_real_methods_test_v4.py --all --extrapolation
"""

import json
import os
import sys
import time
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ============================================================================
# SETUP
# ============================================================================

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

# Import protocols
from hypatiax.experiments.tests.extrapolation_test_protocol import ExtrapolationTestProtocol, REGIMES
from hypatiax.protocols.experiment_protocol_comparative import ComparativeExperimentProtocol

# ============================================================================
# VARIABLE NAME SANITIZER
# ============================================================================


class VariableNameSanitizer:
    """Sanitizes variable names to avoid conflicts with Julia/PySR reserved names."""

    RESERVED_NAMES = {"S", "N", "C", "D", "E", "I", "O"}

    def __init__(self):
        self.forward_mapping: Dict[str, str] = {}
        self.reverse_mapping: Dict[str, str] = {}
        self._conflicts_found = []

    def sanitize(self, variable_names: List[str]) -> Tuple[List[str], bool]:
        """Sanitize variable names to avoid PySR conflicts."""
        sanitized = []
        had_conflicts = False

        for var in variable_names:
            if var in self.RESERVED_NAMES:
                safe_name = f"var_{var}"
                counter = 1
                while safe_name in sanitized or safe_name in variable_names:
                    safe_name = f"var_{var}{counter}"
                    counter += 1

                self.forward_mapping[var] = safe_name
                self.reverse_mapping[safe_name] = var
                self._conflicts_found.append(var)
                sanitized.append(safe_name)
                had_conflicts = True
            else:
                sanitized.append(var)

        return sanitized, had_conflicts

    def restore_expression(self, expression: str) -> str:
        """Restore original variable names in discovered expression."""
        if not self.reverse_mapping or not expression:
            return expression

        restored = expression
        for safe_name in sorted(self.reverse_mapping.keys(), key=len, reverse=True):
            original_name = self.reverse_mapping[safe_name]
            pattern = r"\b" + re.escape(safe_name) + r"\b"
            restored = re.sub(pattern, original_name, restored)

        return restored

    def get_sanitization_log(self) -> str:
        """Return human-readable log of sanitization actions."""
        if not self._conflicts_found:
            return ""

        log_lines = []
        for orig in self._conflicts_found:
            safe = self.forward_mapping[orig]
            log_lines.append(f"{orig}→{safe}")

        return ", ".join(log_lines)


# ============================================================================
# GROUND TRUTH FUNCTION REGISTRY
# ============================================================================


class GroundTruthRegistry:
    """Registry of ground truth functions for extrapolation testing"""

    @staticmethod
    def get_function(
        equation_name: str,
    ) -> Tuple[Callable, Dict[str, Tuple[float, float]]]:
        """Get ground truth function and training ranges"""

        # CHEMISTRY
        if equation_name == "arrhenius":

            def arrhenius(T):
                return 1e11 * np.exp(-80000 / (8.314 * T))

            return arrhenius, {"T": (273, 373)}

        elif equation_name == "henderson_hasselbalch":

            def henderson(A_minus, HA):
                return 6.5 + np.log10(A_minus / (HA + 1e-10))

            return henderson, {"A_minus": (0.1, 2.0), "HA": (0.1, 2.0)}

        elif equation_name == "rate_law":

            def rate_law(A_conc, B_conc):
                return 0.5 * (A_conc**2) * B_conc

            return rate_law, {"A_conc": (0.1, 5.0), "B_conc": (0.1, 5.0)}

        # BIOLOGY
        elif equation_name == "allometric_scaling":

            def allometric(M):
                return 3.5 * (M**0.75)

            return allometric, {"M": (1, 100)}

        elif equation_name == "michaelis_menten":

            def michaelis(S):
                return (50 * S) / (10 + S)

            return michaelis, {"S": (0.1, 50)}

        elif equation_name == "logistic_growth":

            def logistic(N):
                return 0.3 * N * (1 - N / 1000)

            return logistic, {"N": (10, 900)}

        # PHYSICS
        elif equation_name == "kinetic_energy":

            def kinetic(m, v):
                return 0.5 * m * (v**2)

            return kinetic, {"m": (0.1, 10), "v": (0.1, 50)}

        elif equation_name == "gravitational_force":

            def gravity(m1, m2, r):
                return 6.674e-11 * m1 * m2 / (r**2)

            return gravity, {"m1": (1e20, 1e24), "m2": (1e20, 1e24), "r": (1e6, 1e9)}

        elif equation_name == "ideal_gas_law":

            def ideal_gas(n, T, V):
                return n * 8.314 * T / V

            return ideal_gas, {"n": (0.1, 10), "T": (200, 400), "V": (0.01, 1)}

        # DEFI
        elif equation_name == "impermanent_loss":

            def imp_loss(price_ratio):
                return 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1

            return imp_loss, {"price_ratio": (0.5, 2.5)}

        elif equation_name == "price_impact":

            def price_imp(reserve, swap):
                return swap / (reserve + swap)

            return price_imp, {"reserve": (10000, 100000), "swap": (10, 10000)}

        elif equation_name == "constant_product":

            def const_prod(x):
                return 1e6 / x

            return const_prod, {"x": (100, 10000)}

        elif equation_name == "var_95":

            def var95(portfolio, volatility):
                return portfolio * volatility * 1.645

            return var95, {"portfolio": (10000, 1000000), "volatility": (0.01, 0.05)}

        elif equation_name == "liquidation_long":

            def liq_long(entry_price, leverage):
                return entry_price * (1 - 1 / (leverage * 0.8))

            return liq_long, {"entry_price": (30000, 50000), "leverage": (2, 10)}

        elif equation_name == "portfolio_var":

            def port_var(var1, var2, rho):
                return np.sqrt(var1**2 + var2**2 + 2 * rho * var1 * var2)

            return port_var, {
                "var1": (5000, 50000),
                "var2": (3000, 30000),
                "rho": (-0.5, 0.9),
            }

        else:
            raise ValueError(f"Unknown equation: {equation_name}")


# ============================================================================
# RESULT DATA STRUCTURE
# ============================================================================


@dataclass
class TestResult:
    """Standardized test result with extrapolation metrics"""

    method: str
    test_name: str
    domain: str
    success: bool
    r2: float
    rmse: float
    time: float
    formula: str = "N/A"
    error: Optional[str] = None
    metadata: Dict = None

    # Extrapolation metrics
    extrapolation_errors: Dict[str, float] = field(default_factory=dict)
    extrapolation_r2: Dict[str, float] = field(default_factory=dict)

    # NEW: Cache for method-specific data needed for prediction
    _prediction_cache: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        result_dict = {
            "method": self.method,
            "test_name": self.test_name,
            "domain": self.domain,
            "success": self.success,
            "r2": float(self.r2),
            "rmse": float(self.rmse),
            "time": float(self.time),
            "formula": self.formula,
            "error": self.error,
            "metadata": self.metadata or {},
            "extrapolation_errors": {
                k: float(v) for k, v in self.extrapolation_errors.items()
            },
            "extrapolation_r2": {k: float(v) for k, v in self.extrapolation_r2.items()},
        }
        # Don't serialize _prediction_cache (not JSON serializable)
        return result_dict


# ============================================================================
# METHOD WRAPPERS - FIXED PREDICTION METHODS
# ============================================================================


class PureLLMMethodWrapper:
    """Wrapper for baseline_pure_llm.py - FIXED PREDICTION"""

    def __init__(self):
        self.name = "Pure LLM"

        try:
            from hypatiax.core.generation.baseline_pure_llm import PureLLMBaseline

            self.baseline = PureLLMBaseline()
            print(f"✅ {self.name} loaded")
        except Exception as e:
            print(f"❌ {self.name} failed to load: {e}")
            self.baseline = None

    def run(
        self, test_name, description, domain, X, y, var_names, metadata, verbose=False
    ):
        if not self.baseline:
            return TestResult(
                method=self.name,
                test_name=test_name,
                domain=domain,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                time=0.0,
                error="Method not available",
            )

        start = time.time()

        try:
            # Generate formula
            result = self.baseline.generate_formula(
                description=description,
                domain=domain,
                variable_names=var_names,
                metadata=metadata,
            )

            if "error" in result:
                return TestResult(
                    method=self.name,
                    test_name=test_name,
                    domain=domain,
                    success=False,
                    r2=0.0,
                    rmse=float("inf"),
                    time=time.time() - start,
                    error=result["error"],
                )

            # Test accuracy
            metrics = self.baseline.test_formula_accuracy(result, X, y, verbose=False)
            # Compile function for reuse (EXTRAPOLATION FIX)
            compiled_func = self.baseline.compile_formula(result)

            # Create TestResult
            test_result = TestResult(
                method=self.name,
                test_name=test_name,
                domain=domain,
                success=metrics.get("success", False),
                r2=metrics.get("r2", 0.0),
                rmse=metrics.get("rmse", float("inf")),
                time=time.time() - start,
                formula=result.get("formula", "N/A")[:80],
            )

            test_result._prediction_cache["compiled_function"] = compiled_func
            test_result._prediction_cache["variable_names"] = var_names

            if compiled_func is None and verbose:
                print("⚠️ Pure LLM produced no reusable callable")

            if verbose:
                print(
                    f"\n      [DEBUG] Compiled function exists: {compiled_func is not None}"
                )
                print(f"      [DEBUG] Function type: {type(compiled_func)}")

            test_result._prediction_cache["compiled_function"] = compiled_func
            test_result._prediction_cache["variable_names"] = var_names

            return test_result

        except Exception as e:
            return TestResult(
                method=self.name,
                test_name=test_name,
                domain=domain,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                time=time.time() - start,
                error=str(e)[:100],
            )

    def _predict(self, X, result, metadata):
        """Make predictions for extrapolation testing - FIXED!"""

        # DEBUG: Check what's in the cache
        print(f"\n      [DEBUG] Cache keys: {list(result._prediction_cache.keys())}")

        # Get function from result's prediction cache
        cached_function = result._prediction_cache.get("compiled_function")

        print(f"      [DEBUG] Cached function: {cached_function is not None}")

        if not cached_function:
            raise ValueError("No cached function available in result")

        try:
            num_vars = X.shape[1]

            # Handle different numbers of variables
            if num_vars == 1:
                predictions = cached_function(X[:, 0])
            elif num_vars == 2:
                predictions = cached_function(X[:, 0], X[:, 1])
            elif num_vars == 3:
                predictions = cached_function(X[:, 0], X[:, 1], X[:, 2])
            else:
                # For 4+ variables, unpack dynamically
                predictions = cached_function(*[X[:, i] for i in range(num_vars)])

            # Ensure output is numpy array
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            # Flatten if needed
            if predictions.ndim > 1:
                predictions = predictions.flatten()

            # Validate output
            if len(predictions) != len(X):
                raise ValueError(
                    f"Prediction length mismatch: {len(predictions)} != {len(X)}"
                )

            return predictions

        except Exception as e:
            print(f"\n      [DEBUG] Pure LLM prediction failed: {e}")
            raise


class NeuralNetworkMethodWrapper:
    """Wrapper for baseline_neural_network.py - FIXED PREDICTION"""

    def __init__(self):
        self.name = "Neural Network"

        try:
            from hypatiax.core.training.baseline_neural_network import (
                train_and_evaluate,
            )

            self.train_eval = train_and_evaluate
            print(f"✅ {self.name} loaded")
        except Exception as e:
            print(f"❌ {self.name} failed to load: {e}")
            self.train_eval = None

    def run(
        self, test_name, description, domain, X, y, var_names, metadata, verbose=False
    ):
        if not self.train_eval:
            return TestResult(
                method=self.name,
                test_name=test_name,
                domain=domain,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                time=0.0,
                error="Method not available",
            )

        start = time.time()

        try:
            result = self.train_eval(X, y, description, domain, metadata, epochs=200)

            eval_data = result.get("evaluation", {})

            # Create TestResult
            test_result = TestResult(
                method=self.name,
                test_name=test_name,
                domain=domain,
                success=eval_data.get("success", False),
                r2=eval_data.get("r2", 0.0),
                rmse=eval_data.get("rmse", float("inf")),
                time=time.time() - start,
                formula="Neural Network",
            )

            # Store model and scalers in result's prediction cache
            test_result._prediction_cache["model"] = result.get("model")
            test_result._prediction_cache["scaler_X"] = result.get("scaler_X")
            test_result._prediction_cache["scaler_y"] = result.get("scaler_y")

            return test_result

        except Exception as e:
            return TestResult(
                method=self.name,
                test_name=test_name,
                domain=domain,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                time=time.time() - start,
                error=str(e)[:100],
            )

    def _predict(self, X, result, metadata):
        """Make predictions for extrapolation testing - FIXED!"""

        # Get model and scalers from result's prediction cache
        model = result._prediction_cache.get("model")
        scaler_X = result._prediction_cache.get("scaler_X")
        scaler_y = result._prediction_cache.get("scaler_y")

        if not model:
            raise ValueError("No cached model available in result")

        try:
            import torch

            # Scale input
            if scaler_X:
                X_scaled = scaler_X.transform(X)
            else:
                X_scaled = X

            # Predict
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                y_pred_scaled = model(X_tensor).numpy().flatten()

            # Inverse scale output
            if scaler_y:
                y_pred = scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()
            else:
                y_pred = y_pred_scaled

            # Validate output
            if len(y_pred) != len(X):
                raise ValueError(
                    f"Prediction length mismatch: {len(y_pred)} != {len(X)}"
                )

            return y_pred

        except Exception as e:
            print(f"\n      [DEBUG] Neural Network prediction failed: {e}")
            raise


class HybridSystemV40Wrapper:
    """Wrapper for hybrid_system_v43 - FIXED PREDICTION WITH VARIABLE RESOLUTION"""

    def __init__(self):
        self.name = "Hybrid System v40"

        try:
            from hypatiax.tools.symbolic.hybrid_system_v43 import HybridDiscoverySystem
            from hypatiax.tools.symbolic.symbolic_engine import DiscoveryConfig

            self.HybridDiscoverySystem = HybridDiscoverySystem
            self.DiscoveryConfig = DiscoveryConfig
            print(f"✅ {self.name} loaded (with sanitization)")
        except Exception as e:
            print(f"❌ {self.name} failed to load: {e}")
            self.HybridDiscoverySystem = None
            self.DiscoveryConfig = None

    def run(
        self, test_name, description, domain, X, y, var_names, metadata, verbose=False
    ):
        if not self.HybridDiscoverySystem:
            return TestResult(
                method=self.name,
                test_name=test_name,
                domain=domain,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                time=0.0,
                error="Method not available",
            )

        start = time.time()

        try:
            # Apply variable name sanitization
            sanitizer = VariableNameSanitizer()
            sanitized_vars, had_conflicts = sanitizer.sanitize(var_names)

            if had_conflicts and verbose:
                print(f"\n⚠️  Variable sanitization: {sanitizer.get_sanitization_log()}")

            # Create config
            config = self.DiscoveryConfig(
                niterations=50,
                populations=12,
                enable_auto_configuration=True,
                auto_config_correlation_threshold=0.2,
            )

            # Create hybrid system
            hybrid = self.HybridDiscoverySystem(
                domain=domain,
                discovery_config=config,
                enable_auto_config=True,
                max_retries=5,
                enable_physics_fallback=False,
            )

            # Run discovery with SANITIZED variable names
            result = hybrid.discover_validate_interpret(
                X=X,
                y=y,
                variable_names=sanitized_vars,
                variable_descriptions=metadata.get("variable_descriptions", {}),
                variable_units=metadata.get("units", {}),
                description=description,
                equation_name=metadata.get("equation_name", test_name),
                validate_first=True,
            )

            # Extract results
            discovery = result.get("discovery", {})
            validation = result.get("validation", {})

            r2 = discovery.get("r2_score", 0.0)
            val_score = validation.get("total_score", 0.0)

            # Success criteria
            success = (r2 > 0.99 and val_score > 30.0) or (
                r2 > 0.95 and val_score > 80.0
            )

            # Get expression
            expression = discovery.get("expression")

            # Restore original variable names in expression
            if expression and had_conflicts:
                expression_str = str(expression)
                restored_str = sanitizer.restore_expression(expression_str)
                if verbose:
                    print(f"✅ Restored: {expression_str} → {restored_str}")
                expression = restored_str

            # Create TestResult
            test_result = TestResult(
                method=self.name,
                test_name=test_name,
                domain=domain,
                success=success,
                r2=r2,
                rmse=discovery.get("rmse", float("inf")),
                time=time.time() - start,
                formula=str(expression)[:80] if expression else "N/A",
            )

            # Store expression and variable names in result's prediction cache
            test_result._prediction_cache["expression"] = expression
            test_result._prediction_cache["variable_names"] = (
                var_names  # ORIGINAL names
            )
            test_result._prediction_cache["sanitizer"] = sanitizer

            return test_result

        except Exception as e:
            return TestResult(
                method=self.name,
                test_name=test_name,
                domain=domain,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                time=time.time() - start,
                error=str(e)[:100],
            )

    def _predict(self, X, result, metadata):
        """Make predictions for extrapolation testing - FULLY FIXED!"""

        # Get expression and variable names from result's prediction cache
        expression = result._prediction_cache.get("expression")
        var_names = result._prediction_cache.get("variable_names")

        if not expression:
            raise ValueError("No cached expression available in result")

        import sympy as sp

        try:
            # Use the variable names from the cache
            if not var_names or len(var_names) != X.shape[1]:
                # Fallback: try metadata
                var_names = list(metadata.get("variable_descriptions", {}).keys())

            if not var_names or len(var_names) != X.shape[1]:
                # Last resort: extract from expression
                expr_str = str(expression)
                potential_vars = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr_str)
                # Filter out function names
                var_names = sorted(
                    set(
                        v
                        for v in potential_vars
                        if v
                        not in [
                            "exp",
                            "log",
                            "sqrt",
                            "sin",
                            "cos",
                            "tan",
                            "log10",
                            "abs",
                            "max",
                            "min",
                        ]
                    )
                )[: X.shape[1]]

            if len(var_names) != X.shape[1]:
                raise ValueError(
                    f"Variable count mismatch: {len(var_names)} names for {X.shape[1]} columns"
                )

            # Create sympy symbols using the variable names
            symbols_dict = {name: sp.Symbol(name) for name in var_names}

            # Parse expression
            if isinstance(expression, str):
                expr_parsed = sp.sympify(expression, locals=symbols_dict)
            else:
                expr_parsed = expression

            # Lambdify with ordered symbols
            func = sp.lambdify(
                list(symbols_dict.values()), expr_parsed, modules=["numpy", "math"]
            )

            # Make predictions
            if X.shape[1] == 1:
                predictions = func(X[:, 0])
            else:
                predictions = func(*[X[:, i] for i in range(X.shape[1])])

            # Ensure array format
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            if predictions.ndim > 1:
                predictions = predictions.flatten()

            # Validate output
            if len(predictions) != len(X):
                raise ValueError(
                    f"Prediction length mismatch: {len(predictions)} != {len(X)}"
                )

            return predictions

        except Exception as e:
            print(f"\n      [DEBUG] Hybrid prediction failed: {e}")
            print(f"      Expression: {expression}")
            print(f"      Variable names: {var_names}")
            print(f"      X shape: {X.shape}")
            raise


# ============================================================================
# TEST RUNNER
# ============================================================================


class StandaloneTestRunner:
    """Run tests across all domains with extrapolation evaluation"""

    def __init__(self, enable_extrapolation: bool = False):
        print("\n" + "=" * 80)
        title = "STANDALONE TEST SUITE v4 - ALL DOMAINS"
        if enable_extrapolation:
            title += " + EXTRAPOLATION (FIXED!)"
        print(title.center(80))
        print("=" * 80)

        self.enable_extrapolation = enable_extrapolation

        # Initialize REAL methods via wrappers
        print("\n📦 Loading real method implementations...")
        self.methods = [
            PureLLMMethodWrapper(),
            NeuralNetworkMethodWrapper(),
            HybridSystemV40Wrapper(),
        ]

        # Filter out failed methods
        self.methods = [m for m in self.methods if hasattr(m, "run")]

        print(f"\n✅ Loaded {len(self.methods)} methods")
        for m in self.methods:
            print(f"   • {m.name}")

        if enable_extrapolation:
            print("\n🔬 Extrapolation testing: ENABLED (v4 FIXES APPLIED)")
            print("   Regimes: Near (1.2×), Medium (2×), Far (5×)")
            print("   Enhanced debugging and error handling enabled")

        print("=" * 80 + "\n")

        self.results = []
        self.domain_stats = {}

    def run_all_domains(
        self,
        protocol: ComparativeExperimentProtocol,
        num_samples: int = 200,
        verbose: bool = True,
    ):
        """Run tests across all domains"""

        all_domains = protocol.get_all_domains()
        total_tests = sum(
            len(protocol.load_test_data(d, num_samples=10)) for d in all_domains
        )

        print(f"\n📊 Running {total_tests} tests across {len(all_domains)} domains\n")

        test_counter = 0

        for domain_idx, domain in enumerate(all_domains, 1):
            print(f"\n{'='*80}")
            print(
                f"DOMAIN {domain_idx}/{len(all_domains)}: {domain.upper()}".center(80)
            )
            print(f"{'='*80}")

            tests = protocol.load_test_data(domain, num_samples=num_samples)

            for test_idx, (desc, X, y, var_names, metadata) in enumerate(tests, 1):
                test_counter += 1

                if verbose:
                    print(
                        f"\n[{test_counter}/{total_tests}] {metadata['equation_name']}"
                    )

                self.run_test_with_extrapolation(
                    test_name=metadata["equation_name"],
                    description=desc,
                    domain=domain,
                    X_train=X,
                    y_train=y,
                    var_names=var_names,
                    metadata=metadata,
                    verbose=verbose,
                )

            # Domain summary
            self._print_domain_summary(domain)

        print(f"\n{'='*80}")
        print("ALL DOMAINS COMPLETE".center(80))
        print(f"{'='*80}\n")

    def run_test_with_extrapolation(
        self,
        test_name: str,
        description: str,
        domain: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = True,
    ):
        """Run single test with optional extrapolation evaluation"""

        # Get ground truth function if extrapolation enabled
        extrapolation_data = {}
        if self.enable_extrapolation:
            try:
                ground_truth_func, training_ranges = GroundTruthRegistry.get_function(
                    metadata.get("equation_name", test_name)
                )

                if verbose:
                    print(f"   🎯 Generating extrapolation data...", end=" ")

                # Generate extrapolation data for each regime
                for regime in REGIMES:
                    X_extrap, y_extrap = self._generate_multivariate_extrapolation(
                        ground_truth_func,
                        var_names,
                        training_ranges,
                        regime,
                        num_samples=100,
                    )
                    extrapolation_data[regime.name] = (X_extrap, y_extrap)

                if verbose:
                    print("✓")

            except ValueError as e:
                if verbose:
                    print(f"   ⚠️  No ground truth available - skipping extrapolation")

        # Run tests
        test_results = {}

        for method in self.methods:
            if verbose:
                print(f"   Running {method.name}...", end=" ", flush=True)

            # Train on training data
            result = method.run(
                test_name=test_name,
                description=description,
                domain=domain,
                X=X_train,
                y=y_train,
                var_names=var_names,
                metadata=metadata,
                verbose=False,
            )

            # Evaluate extrapolation if enabled and data available
            if self.enable_extrapolation and extrapolation_data and result.success:
                self._evaluate_extrapolation(
                    method, result, extrapolation_data, metadata, verbose=verbose
                )

            test_results[method.name] = result

            # Print result
            if verbose:
                if result.success:
                    extrap_str = ""
                    if (
                        self.enable_extrapolation
                        and "medium" in result.extrapolation_errors
                    ):
                        err = result.extrapolation_errors["medium"]
                        if err < 10000:  # Valid error
                            extrap_str = f", Extrap: {err:.1f}%"
                    print(f"R²={result.r2:.4f}{extrap_str} ✓")
                else:
                    error = result.error[:30] if result.error else "Failed"
                    print(f"✗ {error}")

        # Store results
        self.results.append(
            {
                "test_name": test_name,
                "description": description,
                "domain": domain,
                "results": {name: res.to_dict() for name, res in test_results.items()},
            }
        )

    def _generate_multivariate_extrapolation(
        self,
        ground_truth_func: Callable,
        var_names: List[str],
        training_ranges: Dict[str, Tuple[float, float]],
        regime,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate extrapolation data for multivariate functions"""

        num_vars = len(var_names)
        X_extrap = np.zeros((num_samples, num_vars))

        # Generate extrapolated values for each variable
        for i, var_name in enumerate(var_names):
            x_min, x_max = training_ranges[var_name]

            # Extrapolation range
            extrap_min = x_max * regime.multiplier
            extrap_max = x_max * regime.multiplier * 1.5

            X_extrap[:, i] = np.random.uniform(extrap_min, extrap_max, num_samples)

        # Compute ground truth
        if num_vars == 1:
            y_extrap = ground_truth_func(X_extrap[:, 0])
        elif num_vars == 2:
            y_extrap = ground_truth_func(X_extrap[:, 0], X_extrap[:, 1])
        elif num_vars == 3:
            y_extrap = ground_truth_func(X_extrap[:, 0], X_extrap[:, 1], X_extrap[:, 2])
        else:
            y_extrap = ground_truth_func(*[X_extrap[:, i] for i in range(num_vars)])

        return X_extrap, y_extrap

    def _evaluate_extrapolation(
        self,
        method,
        result: TestResult,
        extrapolation_data: Dict,
        metadata: Dict,
        verbose: bool,
    ):
        """Evaluate method on extrapolation data - FIXED WITH DEBUGGING"""

        rmse_train = result.rmse

        for regime_name, (X_extrap, y_extrap) in extrapolation_data.items():
            try:
                # ADD THIS DEBUG OUTPUT
                if verbose:
                    print(f"\n      [DEBUG] Testing {regime_name} extrapolation...")
                    print(
                        f"      [DEBUG] Cache keys: {list(result._prediction_cache.keys())}"
                    )

                # Get predictions on extrapolation data
                y_pred = method._predict(X_extrap, result, metadata)

                # Validate predictions
                if y_pred is None or len(y_pred) == 0:
                    raise ValueError("Prediction returned empty array")

                if np.any(np.isnan(y_pred)):
                    raise ValueError(
                        f"Prediction contains {np.sum(np.isnan(y_pred))} NaN values"
                    )

                if np.any(np.isinf(y_pred)):
                    raise ValueError(
                        f"Prediction contains {np.sum(np.isinf(y_pred))} Inf values"
                    )

                # Calculate RMSE
                rmse_extrap = np.sqrt(np.mean((y_extrap - y_pred) ** 2))

                # Calculate extrapolation error as percentage
                if rmse_train > 1e-10 and rmse_extrap < 1e10:
                    extrap_error = rmse_extrap / rmse_train
                else:
                    extrap_error = float("inf")

                # Calculate R²
                ss_res = np.sum((y_extrap - y_pred) ** 2)
                ss_tot = np.sum((y_extrap - np.mean(y_extrap)) ** 2)

                if ss_tot > 1e-10:
                    r2_extrap = 1 - (ss_res / ss_tot)
                else:
                    r2_extrap = float("-inf")

                # Store metrics
                result.extrapolation_errors[regime_name] = extrap_error
                result.extrapolation_r2[regime_name] = r2_extrap

            except Exception as e:
                # Log the error for debugging
                if verbose:
                    print(f"\n      [EXTRAP ERROR] {regime_name}: {str(e)[:60]}")

                result.extrapolation_errors[regime_name] = float("inf")
                result.extrapolation_r2[regime_name] = float("-inf")

    def _print_domain_summary(self, domain: str):
        """Print summary for completed domain"""

        domain_results = [r for r in self.results if r["domain"] == domain]

        if not domain_results:
            return

        print(f"\n{'─'*80}")
        print(f"DOMAIN SUMMARY: {domain.upper()}")
        print(f"{'─'*80}")

        # Calculate stats per method
        method_stats = {}

        for test in domain_results:
            for method_name, result in test["results"].items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        "successes": 0,
                        "failures": 0,
                        "r2_scores": [],
                        "times": [],
                        "extrap_errors": [],
                    }

                stats = method_stats[method_name]

                if result["success"]:
                    stats["successes"] += 1
                    stats["r2_scores"].append(result["r2"])
                    stats["times"].append(result["time"])

                    if self.enable_extrapolation and "medium" in result.get(
                        "extrapolation_errors", {}
                    ):
                        err = result["extrapolation_errors"]["medium"]
                        if err < 10000:
                            stats["extrap_errors"].append(err)
                else:
                    stats["failures"] += 1

        # Print stats
        total_tests = len(domain_results)

        for method, stats in method_stats.items():
            success_rate = (
                (stats["successes"] / total_tests * 100) if total_tests > 0 else 0
            )

            avg_r2 = np.mean(stats["r2_scores"]) if stats["r2_scores"] else 0

            extrap_str = ""
            if self.enable_extrapolation and stats["extrap_errors"]:
                avg_extrap = np.mean(stats["extrap_errors"])
                extrap_str = f", Avg Extrap: {avg_extrap:6.1f}%"

            print(
                f"  {method:<30} Success: {stats['successes']}/{total_tests} ({success_rate:5.1f}%), "
                f"Avg R²: {avg_r2:.4f}{extrap_str}"
            )

        # Store for final summary
        self.domain_stats[domain] = method_stats

    def print_summary(self):
        """Print overall summary with extrapolation stats"""

        if not self.results:
            print("\n⚠️  No tests run!")
            return

        print(f"\n{'='*80}")
        print("FINAL SUMMARY - ALL DOMAINS".center(80))
        print(f"{'='*80}")

        total_tests = len(self.results)
        print(f"\n📊 Total tests: {total_tests}")

        # Collect overall statistics
        method_stats = {}

        for test in self.results:
            for method_name, result in test["results"].items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        "r2_scores": [],
                        "times": [],
                        "successes": 0,
                        "failures": 0,
                        "wins": 0,
                        "extrap_near": [],
                        "extrap_medium": [],
                        "extrap_far": [],
                    }

                stats = method_stats[method_name]

                if result["success"]:
                    stats["successes"] += 1
                    stats["r2_scores"].append(result["r2"])
                    stats["times"].append(result["time"])

                    # Collect extrapolation errors
                    if self.enable_extrapolation:
                        for regime in ["near", "medium", "far"]:
                            if regime in result.get("extrapolation_errors", {}):
                                err = result["extrapolation_errors"][regime]
                                if err < 10000:  # Filter catastrophic failures
                                    stats[f"extrap_{regime}"].append(err)
                else:
                    stats["failures"] += 1

        # Count wins (best R² per test)
        for test in self.results:
            best_r2 = (
                max(r["r2"] for r in test["results"].values() if r["success"])
                if any(r["success"] for r in test["results"].values())
                else -1
            )

            for method_name, result in test["results"].items():
                if result["success"] and result["r2"] == best_r2:
                    method_stats[method_name]["wins"] += 1
                    break

        # Print method comparison
        print(f"\n{'─'*80}")
        print("METHOD COMPARISON")
        print(f"{'─'*80}")

        print(f"\n🏆 Success Rate:")
        for method in sorted(
            method_stats.keys(),
            key=lambda x: method_stats[x]["successes"],
            reverse=True,
        ):
            stats = method_stats[method]
            success_rate = (
                (stats["successes"] / total_tests * 100) if total_tests > 0 else 0
            )
            bar = "█" * int(success_rate / 5)
            print(
                f"   {method:<35} {stats['successes']:>3}/{total_tests}  ({success_rate:>5.1f}%) {bar}"
            )

        print(f"\n📈 Average R² (successful tests):")
        for method in sorted(
            method_stats.keys(),
            key=lambda x: (
                np.mean(method_stats[x]["r2_scores"])
                if method_stats[x]["r2_scores"]
                else -1
            ),
            reverse=True,
        ):
            stats = method_stats[method]
            if stats["r2_scores"]:
                avg_r2 = np.mean(stats["r2_scores"])
                std_r2 = np.std(stats["r2_scores"])
                print(f"   {method:<35} {avg_r2:.4f} ± {std_r2:.4f}")
            else:
                print(f"   {method:<35} N/A (no successful runs)")

        print(f"\n⏱️  Average Time:")
        for method in sorted(
            method_stats.keys(),
            key=lambda x: (
                np.mean(method_stats[x]["times"])
                if method_stats[x]["times"]
                else float("inf")
            ),
        ):
            stats = method_stats[method]
            if stats["times"]:
                avg_time = np.mean(stats["times"])
                print(f"   {method:<35} {avg_time:>6.1f}s")

        print(f"\n🥇 Wins (best R² per test):")
        for method in sorted(
            method_stats.keys(), key=lambda x: method_stats[x]["wins"], reverse=True
        ):
            stats = method_stats[method]
            win_rate = (stats["wins"] / total_tests * 100) if total_tests > 0 else 0
            print(
                f"   {method:<35} {stats['wins']:>3}/{total_tests}  ({win_rate:>5.1f}%)"
            )

        # Extrapolation summary
        if self.enable_extrapolation:
            print(f"\n{'─'*80}")
            print("EXTRAPOLATION PERFORMANCE")
            print(f"{'─'*80}")

            for regime in ["near", "medium", "far"]:
                regime_mult = next(r.multiplier for r in REGIMES if r.name == regime)
                print(f"\n{regime.capitalize()} Extrapolation ({regime_mult}×):")

                for method in sorted(
                    method_stats.keys(),
                    key=lambda x: (
                        np.mean(method_stats[x][f"extrap_{regime}"])
                        if method_stats[x][f"extrap_{regime}"]
                        else float("inf")
                    ),
                ):
                    stats = method_stats[method]
                    errors = stats[f"extrap_{regime}"]

                    if errors:
                        avg_err = np.mean(errors)
                        std_err = np.std(errors)
                        n_valid = len(errors)
                        n_total = stats["successes"]

                        # Categorize
                        if avg_err < 50:
                            status = "✅ EXCELLENT"
                        elif avg_err < 100:
                            status = "✓ GOOD"
                        elif avg_err < 200:
                            status = "⚠️  MODERATE"
                        elif avg_err < 500:
                            status = "✗ POOR"
                        else:
                            status = "💥 CATASTROPHIC"

                        print(
                            f"   {method:<35} {avg_err:6.1f}% ± {std_err:5.1f}%  {status}  ({n_valid}/{n_total})"
                        )
                    else:
                        print(f"   {method:<35} N/A (0 valid predictions)")

        # Domain breakdown
        print(f"\n{'─'*80}")
        print("PERFORMANCE BY DOMAIN")
        print(f"{'─'*80}")

        for domain, stats in self.domain_stats.items():
            print(f"\n{domain.upper()}:")
            domain_tests = len([r for r in self.results if r["domain"] == domain])

            for method in sorted(stats.keys()):
                method_data = stats[method]
                success_rate = (
                    (method_data["successes"] / domain_tests * 100)
                    if domain_tests > 0
                    else 0
                )
                avg_r2 = (
                    np.mean(method_data["r2_scores"]) if method_data["r2_scores"] else 0
                )

                print(
                    f"   {method:<30} {method_data['successes']}/{domain_tests} ({success_rate:5.1f}%), "
                    f"R²: {avg_r2:.4f}"
                )

        print(f"\n{'='*80}\n")

        # Save results
        self._save_results()

        # Generate Table 1 data
        if self.enable_extrapolation:
            self._generate_table1_data(method_stats)

    def _generate_table1_data(self, method_stats):
        """Generate data formatted for JMLR paper Table 1"""

        print(f"\n{'='*80}")
        print("TABLE 1 DATA FOR JMLR PAPER".center(80))
        print(f"{'='*80}\n")

        print("LaTeX table data:")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print(
            "\\textbf{Method} & \\textbf{Accuracy (R²)} & \\textbf{Extrap. Error} & \\textbf{Correct Form} & \\textbf{Time} \\\\"
        )
        print("\\midrule")

        total_tests = len(self.results)

        for method, stats in sorted(method_stats.items()):
            avg_r2 = np.mean(stats["r2_scores"]) if stats["r2_scores"] else 0
            std_r2 = np.std(stats["r2_scores"]) if stats["r2_scores"] else 0

            # Use medium extrapolation for table
            extrap_errors = stats["extrap_medium"]
            avg_extrap = np.mean(extrap_errors) if extrap_errors else 0

            correct = stats["successes"]
            total = total_tests

            avg_time = np.mean(stats["times"]) if stats["times"] else 0

            print(
                f"{method} & ${avg_r2:.2f} \\pm {std_r2:.2f}$ & {avg_extrap:.0f}\\% & "
                f"{correct}/{total} ({correct/total*100:.1f}\\%) & {avg_time:.1f}s \\\\"
            )

        print("\\bottomrule")
        print("\\end{tabular}")
        print(f"\n{'='*80}\n")

    def _save_results(self):
        """Save results to JSON"""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_extrap_v4" if self.enable_extrapolation else "_v4"
        output_file = output_dir / f"all_domains{suffix}_{timestamp}.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "version": "v4 - FIXED extrapolation predictions",
            "extrapolation_enabled": self.enable_extrapolation,
            "methods": [m.name for m in self.methods],
            "total_tests": len(self.results),
            "domains": list(self.domain_stats.keys()),
            "tests": self.results,
        }

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Results saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone Test Suite v4 - FIXED Extrapolation!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ALL tests WITH extrapolation (FIXED!)
  python standalone_real_methods_test_v4.py --all --extrapolation

  # Run specific domain with extrapolation
  python standalone_real_methods_test_v4.py --domain biology --extrapolation

  # Run single test with extrapolation
  python standalone_real_methods_test_v4.py --test michaelis_menten --extrapolation

WHAT'S NEW IN V4:
  ✅ FIXED: Extrapolation predictions now work properly!
  ✅ Enhanced error handling in all _predict() methods
  ✅ Better variable name resolution for Hybrid System
  ✅ Validation of prediction outputs (NaN/Inf checks)
  ✅ Diagnostic output for debugging failures
  ✅ Proper filtering of invalid extrapolation errors
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run all domains")
    parser.add_argument("--domain", type=str, help="Run specific domain only")
    parser.add_argument("--test", type=str, help="Run single test only")
    parser.add_argument(
        "--extrapolation", action="store_true", help="Enable extrapolation testing"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of training samples (default: 200)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Load protocol
    try:
        protocol = ComparativeExperimentProtocol()
    except ImportError:
        print("❌ Error: experiment_protocol_comparative.py not found")
        return

    # Initialize runner
    runner = StandaloneTestRunner(enable_extrapolation=args.extrapolation)

    # Determine what to run
    if args.all:
        # Run all domains
        runner.run_all_domains(
            protocol=protocol, num_samples=args.samples, verbose=not args.quiet
        )

    elif args.test:
        # Single test
        found = False
        for domain in protocol.get_all_domains():
            tests = protocol.load_test_data(domain, num_samples=args.samples)
            for desc, X, y, var_names, metadata in tests:
                if args.test.lower() in metadata["equation_name"].lower():
                    print(f"\n🔍 Found test: {metadata['equation_name']} in {domain}")
                    runner.run_test_with_extrapolation(
                        test_name=metadata["equation_name"],
                        description=desc,
                        domain=domain,
                        X_train=X,
                        y_train=y,
                        var_names=var_names,
                        metadata=metadata,
                        verbose=not args.quiet,
                    )
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"❌ Test '{args.test}' not found")
            print("\nAvailable tests:")
            for domain in protocol.get_all_domains():
                tests = protocol.load_test_data(domain, num_samples=10)
                print(f"\n  {domain}:")
                for _, _, _, _, meta in tests:
                    print(f"    • {meta['equation_name']}")
            return

    elif args.domain:
        # Specific domain
        if args.domain not in protocol.get_all_domains():
            print(f"❌ Unknown domain: {args.domain}")
            print(f"Available domains: {', '.join(protocol.get_all_domains())}")
            return

        print(f"\n📊 Running domain: {args.domain}")

        tests = protocol.load_test_data(args.domain, num_samples=args.samples)

        for desc, X, y, var_names, metadata in tests:
            runner.run_test_with_extrapolation(
                test_name=metadata["equation_name"],
                description=desc,
                domain=args.domain,
                X_train=X,
                y_train=y,
                var_names=var_names,
                metadata=metadata,
                verbose=not args.quiet,
            )

    else:
        # Show help if no option specified
        print("❌ Please specify --all, --domain, or --test")
        print("\nExamples:")
        print("  python standalone_real_methods_test_v4.py --all --extrapolation")
        print("  python standalone_real_methods_test_v4.py --domain chemistry")
        print(
            "  python standalone_real_methods_test_v4.py --test arrhenius --extrapolation"
        )
        return

    # Print summary
    runner.print_summary()

    print("\n✅ Complete!\n")


if __name__ == "__main__":
    main()
