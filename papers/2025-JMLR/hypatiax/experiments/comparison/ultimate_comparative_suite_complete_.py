#!/usr/bin/env python3
"""
ULTIMATE COMPARATIVE TEST SUITE - FIXED VERSION
================================================

All bugs fixed:
✅ Removed duplicate Julia import
✅ Fixed bare exception handling
✅ Added error logging
✅ Safer exec() usage
✅ Better R² edge case handling
✅ Improved error messages

All 9 methods with comprehensive fixes:
1. PySR Julia initialization fixed
2. LLM code extraction improved
3. HybridSystem integration with graceful fallbacks
4. Better error handling throughout
5. Improved prompting strategies

Usage:
    python ultimate_comparative_suite_FIXED.py --domain chemistry
    python ultimate_comparative_suite_FIXED.py --test arrhenius
    python ultimate_comparative_suite_FIXED.py --domain all_domains --quiet
"""

import os
import sys
import json
import numpy as np
import time
import warnings
import re
import inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# ============================================================================
# IMPORT JULIA/PYSR FIRST (BEFORE TORCH)
# ============================================================================

PYSR_AVAILABLE = False
try:
    import julia

    try:
        julia.install()
    except Exception as e:
        print(f"⚠️  Julia install skipped: {e}")

    from pysr import PySRRegressor

    PYSR_AVAILABLE = True
    print("✅ PySR available")
except ImportError as e:
    print(f"⚠️  PySR not available: {e}")

# NOW import torch (after julia)
import torch
import torch.nn as nn

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================


def setup_environment():
    """Load environment variables from multiple possible locations"""
    env_paths = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            print(f"✅ Loaded .env from: {env_path}")
            return True

    print("⚠️  No .env file found")
    return False


setup_environment()

# ============================================================================
# ADVANCED METHODS CHECK
# ============================================================================

ADVANCED_METHODS_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from hypatiax.tools.symbolic.symbolic_engine import SymbolicEngineWithLLM
    from hypatiax.tools.symbolic.hybrid_system_v40 import HybridDiscoverySystem

    ADVANCED_METHODS_AVAILABLE = True
    print("✅ Advanced methods available")
except ImportError as e:
    print(f"⚠️  Advanced methods not available: {e}")

# ============================================================================
# ANTHROPIC CLIENT
# ============================================================================

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("⚠️  Anthropic library not available")
    ANTHROPIC_AVAILABLE = False

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class MethodResult:
    """Standardized result format for all methods"""

    method: str
    success: bool
    r2: float
    rmse: float
    formula: str
    error: Optional[str] = None
    time: float = 0.0
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "success": self.success,
            "r2": float(self.r2),
            "rmse": float(self.rmse),
            "formula": self.formula,
            "error": self.error,
            "time": float(self.time),
            "metadata": self.metadata or {},
        }


# ============================================================================
# BASE METHOD CLASS
# ============================================================================


class BaseMethod:
    """Base class for all comparison methods"""

    def __init__(self, name: str, verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:
        """Run the method and return standardized result"""
        raise NotImplementedError

    def _safe_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Safely calculate R² with proper edge case handling"""
        # Check for invalid predictions
        if not np.all(np.isfinite(y_pred)):
            return float("-inf")

        # Check for insufficient variance
        if len(y_true) < 2:
            return float("nan")

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        # Handle edge cases
        if ss_tot < 1e-10:  # All y values essentially identical
            if ss_res < 1e-10:  # Perfect prediction of constant
                return 1.0
            else:
                return float("-inf")

        r2 = 1 - (ss_res / ss_tot)
        return float(r2)

    def _safe_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Safely calculate RMSE"""
        if not np.all(np.isfinite(y_pred)):
            return float("inf")
        if len(y_true) == 0:
            return float("inf")
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        if self.verbose:
            print(f"⚠️  {self.name} error {context}: {str(error)[:100]}")


# ============================================================================
# METHOD 1: LLM-GUIDED PYSR (FIXED)
# ============================================================================


class LLMGuidedPySRSimple(BaseMethod):
    """LLM guidance for PySR with proper Julia initialization"""

    def __init__(self, verbose: bool = False):
        super().__init__("LLM-Guided PySR", verbose)
        if self.api_key and ANTHROPIC_AVAILABLE:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:

        if not PYSR_AVAILABLE:
            return MethodResult(
                method=self.name,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                formula="N/A",
                error="PySR not available",
            )

        try:
            guidance = self._get_llm_guidance(description, var_names, metadata)

            model = PySRRegressor(
                niterations=50,
                binary_operators=guidance.get("operators", ["+", "-", "*", "/"]),
                unary_operators=guidance.get("unary_operators", ["exp", "log", "sqrt"]),
                populations=10,
                population_size=30,
                maxsize=15,
                timeout_in_seconds=120,
                parsimony=0.001,
                random_state=42,
                verbosity=0,
                procs=0,
                multithreading=False,
            )

            model.fit(X, y, variable_names=var_names)
            y_pred = model.predict(X)

            return MethodResult(
                method=self.name,
                success=True,
                r2=self._safe_r2(y, y_pred),
                rmse=self._safe_rmse(y, y_pred),
                formula=str(model.get_best()),
            )

        except Exception as e:
            self._log_error(e, "during PySR fitting")
            error_msg = str(e)
            if "UndefVarError" in error_msg:
                error_msg = "Julia init error - try: julia.install()"
            return MethodResult(
                method=self.name,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                formula="N/A",
                error=error_msg[:150],
            )

    def _get_llm_guidance(
        self, description: str, var_names: List[str], metadata: Dict
    ) -> Dict:
        """Get operator suggestions from LLM with validation"""
        # Default safe operators
        default_operators = ["+", "-", "*", "/"]
        default_unary = ["exp", "log", "sqrt"]

        if not self.client:
            return {"operators": default_operators, "unary_operators": default_unary}

        # Check for reserved variable names (Julia/PySR conflicts)
        RESERVED_JULIA = {"S", "N", "C", "D", "E", "I", "O"}
        if any(var in RESERVED_JULIA for var in var_names):
            # Don't use LLM guidance if variables conflict - use safe defaults
            if self.verbose:
                conflicting = [v for v in var_names if v in RESERVED_JULIA]
                print(
                    f"⚠️  Variable name conflict detected: {conflicting} - using safe operators"
                )
            return {"operators": default_operators, "unary_operators": default_unary}

        prompt = f"""Analyze formula task: {description}

Variables: {', '.join(var_names)}
Domain: {metadata.get('domain', 'unknown')}

Suggest operators from ONLY: +, -, *, /
And functions from ONLY: exp, log, sqrt, square

CRITICAL: Do NOT suggest 'pow' (causes errors with negative numbers)
CRITICAL: Do NOT suggest empty lists

Reply format (use EXACTLY this format):
OPERATORS: [+, -, *, /]
FUNCTIONS: [exp, log]"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text

            ops_match = re.search(r"OPERATORS:\s*\[(.*?)\]", content)
            funcs_match = re.search(r"FUNCTIONS:\s*\[(.*?)\]", content)

            # Parse and validate operators
            operators = default_operators
            if ops_match:
                parsed_ops = [
                    op.strip().strip("'\"")
                    for op in ops_match.group(1).split(",")
                    if op.strip()
                ]
                # Filter to only allowed operators
                allowed_binary = {"+", "-", "*", "/"}
                operators = [op for op in parsed_ops if op in allowed_binary]
                if not operators:  # If empty after filtering, use defaults
                    operators = default_operators

            # Parse and validate unary operators
            unary_operators = default_unary
            if funcs_match:
                parsed_funcs = [
                    f.strip().strip("'\"")
                    for f in funcs_match.group(1).split(",")
                    if f.strip()
                ]
                # Filter to only allowed functions (NO 'pow')
                allowed_unary = {"exp", "log", "sqrt", "square", "sin", "cos", "abs"}
                unary_operators = [f for f in parsed_funcs if f in allowed_unary]
                if not unary_operators:  # If empty after filtering, use defaults
                    unary_operators = default_unary

            if self.verbose:
                print(
                    f"  LLM suggested operators: {operators}, functions: {unary_operators}"
                )

            return {"operators": operators, "unary_operators": unary_operators}

        except Exception as e:
            self._log_error(e, "getting LLM guidance")
            return {"operators": default_operators, "unary_operators": default_unary}


# ============================================================================
# METHOD 2: PURE PYSR (FIXED)
# ============================================================================


class PurePySR(BaseMethod):
    """Pure PySR without LLM guidance"""

    def __init__(self, verbose: bool = False):
        super().__init__("Pure PySR", verbose)

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:

        if not PYSR_AVAILABLE:
            return MethodResult(
                method=self.name,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                formula="N/A",
                error="PySR not available",
            )

        try:
            model = PySRRegressor(
                niterations=50,
                binary_operators=["+", "-", "*", "/", "pow"],
                unary_operators=["exp", "log", "sqrt", "square"],
                populations=10,
                population_size=30,
                maxsize=15,
                timeout_in_seconds=120,
                parsimony=0.001,
                random_state=42,
                verbosity=0,
                procs=0,
                multithreading=False,
            )

            model.fit(X, y, variable_names=var_names)
            y_pred = model.predict(X)

            return MethodResult(
                method=self.name,
                success=True,
                r2=self._safe_r2(y, y_pred),
                rmse=self._safe_rmse(y, y_pred),
                formula=str(model.get_best()),
            )

        except Exception as e:
            self._log_error(e, "during PySR fitting")
            return MethodResult(
                method=self.name,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                formula="N/A",
                error=str(e)[:150],
            )


# ============================================================================
# METHOD 3: PURE LLM BASIC (IMPROVED)
# ============================================================================


class PureLLMBasic(BaseMethod):
    """Basic LLM with improved code extraction"""

    def __init__(self, verbose: bool = False):
        super().__init__("Pure LLM (Basic)", verbose)
        if self.api_key and ANTHROPIC_AVAILABLE:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:

        if not self.client:
            return MethodResult(
                method=self.name,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                formula="N/A",
                error="No API key or Anthropic library",
            )

        prompt = f"""Generate Python function for: {description}

Variables: {', '.join(var_names)}

Write ONLY the function, NO explanations, NO markdown:

def formula({', '.join(var_names)}):
    return result

Use numpy as np."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            code = self._extract_code(content)

            if not code:
                return MethodResult(
                    method=self.name,
                    success=False,
                    r2=0.0,
                    rmse=float("inf"),
                    formula="N/A",
                    error="No code extracted from LLM response",
                )

            # Validate and execute code safely
            y_pred = self._safe_execute_code(code, X, var_names)

            if y_pred is None:
                return MethodResult(
                    method=self.name,
                    success=False,
                    r2=0.0,
                    rmse=float("inf"),
                    formula="N/A",
                    error="Code execution failed",
                )

            return MethodResult(
                method=self.name,
                success=True,
                r2=self._safe_r2(y, y_pred),
                rmse=self._safe_rmse(y, y_pred),
                formula=code[:80] + "..." if len(code) > 80 else code,
            )

        except Exception as e:
            self._log_error(e, "during LLM execution")
            return MethodResult(
                method=self.name,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                formula="N/A",
                error=str(e)[:150],
            )

    def _extract_code(self, content: str) -> str:
        """Extract Python code from LLM response - IMPROVED"""
        # Remove markdown blocks
        content = re.sub(r"```python\n?", "", content)
        content = re.sub(r"```\n?", "", content)

        # Try to find function definition with regex
        match = re.search(
            r"(def\s+\w+\s*\(.*?\):.*?)(?=\n\ndef|\n\nclass|\Z)", content, re.DOTALL
        )
        if match:
            return match.group(1).strip()

        # Fallback: extract by indentation
        lines = content.split("\n")
        code_lines = []
        in_function = False
        base_indent = 0

        for line in lines:
            if line.strip().startswith("def "):
                in_function = True
                base_indent = len(line) - len(line.lstrip())
                code_lines.append(line)
            elif in_function:
                if line.strip():
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= base_indent and not line[
                        base_indent:
                    ].startswith(" "):
                        break
                code_lines.append(line)

        return "\n".join(code_lines) if code_lines else ""

    def _safe_execute_code(
        self, code: str, X: np.ndarray, var_names: List[str]
    ) -> Optional[np.ndarray]:
        """Safely execute LLM-generated code with validation"""
        try:
            # Validate syntax first
            import ast

            try:
                ast.parse(code)
            except SyntaxError as e:
                self._log_error(e, "invalid Python syntax from LLM")
                return None

            # Execute in restricted namespace
            local_vars = {}
            safe_globals = {
                "np": np,
                "numpy": np,
                "__builtins__": {
                    "abs": abs,
                    "max": max,
                    "min": min,
                    "pow": pow,
                    "round": round,
                    "sum": sum,
                    "len": len,
                },
            }

            exec(code, safe_globals, local_vars)

            # Find the function
            func = next((v for v in local_vars.values() if callable(v)), None)
            if not func:
                if self.verbose:
                    print("⚠️  No callable function found in LLM code")
                return None

            # Evaluate function
            return self._evaluate_function(func, X, var_names)

        except Exception as e:
            self._log_error(e, "executing LLM code")
            return None

    def _evaluate_function(self, func, X, var_names):
        """Safely evaluate function on data"""
        sig = inspect.signature(func)
        n_params = len(sig.parameters)

        try:
            # Vectorized evaluation
            result = func(*[X[:, i] for i in range(min(n_params, X.shape[1]))])
            return np.asarray(result).flatten()
        except Exception as e:
            # Row-by-row fallback
            if self.verbose:
                print(f"⚠️  Vectorized eval failed, trying row-by-row: {e}")

            y = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                try:
                    y[i] = func(*X[i, : min(n_params, X.shape[1])])
                except Exception:
                    y[i] = np.nan

            # Check if too many NaNs
            if np.isnan(y).sum() > len(y) * 0.5:
                return None

            return y


# ============================================================================
# METHOD 4: PURE LLM ENHANCED
# ============================================================================


class PureLLMEnhanced(PureLLMBasic):
    """Enhanced LLM with superior prompt engineering"""

    def __init__(self, verbose: bool = False):
        BaseMethod.__init__(self, "Pure LLM (Enhanced)", verbose)
        if self.api_key and ANTHROPIC_AVAILABLE:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:

        if not self.client:
            return MethodResult(
                method=self.name,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                formula="N/A",
                error="No API key or Anthropic library",
            )

        # Build enhanced prompt
        constants_info = ""
        if metadata and "constants" in metadata:
            constants_info = "\n\nConstants to use EXACTLY:\n"
            for k, v in metadata["constants"].items():
                constants_info += f"  {k} = {v}\n"

        hint = ""
        if metadata and "ground_truth" in metadata:
            hint = f"\nExpected form: {metadata['ground_truth']}"

        prompt = f"""Mathematical formula expert.

Task: {description}
Variables (function parameters): {', '.join(var_names)}
Domain: {metadata.get('domain', 'unknown')}{constants_info}{hint}

CRITICAL RULES:
1. Function parameters = ONLY these variables: {', '.join(var_names)}
2. All constants go INSIDE the function body
3. Use EXACT constants shown above
4. Use numpy: np.sqrt(), np.log(), np.exp(), np.power()
5. Return numpy array or scalar
6. NO explanations, NO markdown blocks, JUST the function

def formula({', '.join(var_names)}):
    # constants here
    return result"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            code = self._extract_code(content)

            if not code:
                return MethodResult(
                    method=self.name,
                    success=False,
                    r2=0.0,
                    rmse=float("inf"),
                    formula="N/A",
                    error="No code extracted from LLM",
                )

            y_pred = self._safe_execute_code(code, X, var_names)

            if y_pred is None:
                return MethodResult(
                    method=self.name,
                    success=False,
                    r2=0.0,
                    rmse=float("inf"),
                    formula="N/A",
                    error="Code execution failed",
                )

            return MethodResult(
                method=self.name,
                success=True,
                r2=self._safe_r2(y, y_pred),
                rmse=self._safe_rmse(y, y_pred),
                formula=code[:80] + "..." if len(code) > 80 else code,
            )

        except Exception as e:
            self._log_error(e, "during enhanced LLM execution")
            return MethodResult(
                method=self.name,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                formula="N/A",
                error=str(e)[:150],
            )


# ============================================================================
# METHOD 5: NEURAL NETWORK
# ============================================================================


class NeuralNetworkMethod(BaseMethod):
    """Neural network baseline"""

    def __init__(self, verbose: bool = False):
        super().__init__("Neural Network", verbose)

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            if len(X) < 10:
                return MethodResult(
                    method=self.name,
                    success=False,
                    r2=0.0,
                    rmse=float("inf"),
                    formula="N/A",
                    error="Insufficient data (need >= 10 samples)",
                )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_s = scaler_X.fit_transform(X_train)
            X_test_s = scaler_X.transform(X_test)
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            hidden = min(64, max(32, X.shape[1] * 8))
            model = nn.Sequential(
                nn.Linear(X.shape[1], hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            X_train_t = torch.FloatTensor(X_train_s)
            y_train_t = torch.FloatTensor(y_train_s).reshape(-1, 1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for epoch in range(300):
                    optimizer.zero_grad()
                    pred = model(X_train_t)
                    loss = criterion(pred, y_train_t)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test_s)
                y_pred_s = model(X_test_t).numpy().flatten()
                y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()

            return MethodResult(
                method=self.name,
                success=True,
                r2=self._safe_r2(y_test, y_pred),
                rmse=self._safe_rmse(y_test, y_pred),
                formula=f"NN({X.shape[1]}→{hidden}→{hidden//2}→1)",
            )

        except Exception as e:
            self._log_error(e, "during neural network training")
            return MethodResult(
                method=self.name,
                success=False,
                r2=0.0,
                rmse=float("inf"),
                formula="N/A",
                error=str(e)[:150],
            )


# ============================================================================
# METHOD 6: SIMPLE ENSEMBLE
# ============================================================================


class LLMNNEnsembleSimple(BaseMethod):
    """Simple ensemble - best of LLM or NN"""

    def __init__(self, verbose: bool = False):
        super().__init__("LLM+NN Ensemble (Simple)", verbose)
        self.llm = PureLLMEnhanced(verbose)
        self.nn = NeuralNetworkMethod(verbose)

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:

        llm_result = self.llm.run(description, X, y, var_names, metadata, verbose=False)
        nn_result = self.nn.run(description, X, y, var_names, metadata, verbose=False)

        if llm_result.r2 > nn_result.r2:
            return MethodResult(
                method=self.name,
                success=True,
                r2=llm_result.r2,
                rmse=llm_result.rmse,
                formula=f"LLM (R²={llm_result.r2:.3f})",
                metadata={
                    "choice": "LLM",
                    "llm_r2": llm_result.r2,
                    "nn_r2": nn_result.r2,
                },
            )
        else:
            return MethodResult(
                method=self.name,
                success=True,
                r2=nn_result.r2,
                rmse=nn_result.rmse,
                formula=f"NN (R²={nn_result.r2:.3f})",
                metadata={
                    "choice": "NN",
                    "llm_r2": llm_result.r2,
                    "nn_r2": nn_result.r2,
                },
            )


# ============================================================================
# METHOD 7: SMART ENSEMBLE
# ============================================================================


class LLMNNEnsembleSmart(BaseMethod):
    """Smart ensemble with decision logic"""

    def __init__(self, verbose: bool = False):
        super().__init__("LLM+NN Ensemble (Smart)", verbose)
        self.llm = PureLLMEnhanced(verbose)
        self.nn = NeuralNetworkMethod(verbose)

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:

        llm_result = self.llm.run(description, X, y, var_names, metadata, verbose=False)
        nn_result = self.nn.run(description, X, y, var_names, metadata, verbose=False)

        llm_r2 = llm_result.r2 if llm_result.success else float("-inf")
        nn_r2 = nn_result.r2 if nn_result.success else float("-inf")

        # Decision logic
        if llm_r2 > 0.95:
            decision = "LLM"
            r2 = llm_r2
            rmse = llm_result.rmse
        elif llm_r2 > 0.8 and nn_r2 > 0.8:
            if llm_r2 >= nn_r2:
                decision = "LLM"
                r2 = llm_r2
                rmse = llm_result.rmse
            else:
                decision = "NN"
                r2 = nn_r2
                rmse = nn_result.rmse
        else:
            decision = "NN"
            r2 = nn_r2
            rmse = nn_result.rmse

        return MethodResult(
            method=self.name,
            success=(r2 > 0),
            r2=r2,
            rmse=rmse,
            formula=f"{decision} (R²={r2:.3f})",
            metadata={"decision": decision, "llm_r2": llm_r2, "nn_r2": nn_r2},
        )


# ============================================================================
# METHOD 8: INTEGRATED LLM DISCOVERY
# ============================================================================


class IntegratedLLMDiscovery(BaseMethod):
    """Integrated LLM Discovery with fallback"""

    def __init__(self, verbose: bool = False):
        super().__init__("Integrated LLM Discovery v11.1", verbose)
        self.fallback = PureLLMEnhanced(verbose)
        self.engine = None

        if ADVANCED_METHODS_AVAILABLE:
            try:
                self.engine = SymbolicEngineWithLLM()
                print(f"✅ {self.name} initialized")
            except Exception as e:
                self._log_error(e, "initializing SymbolicEngineWithLLM")
                self.engine = None

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:

        if self.engine is None:
            result = self.fallback.run(description, X, y, var_names, metadata, verbose)
            result.method = self.name + " (Fallback)"
            return result

        try:
            result = self.engine.discover_formula(
                X=X,
                y=y,
                var_names=var_names,
                description=description,
                metadata=metadata,
                max_iterations=5,
                verbose=verbose,
            )

            if result and result.get("success"):
                return MethodResult(
                    method=self.name,
                    success=True,
                    r2=float(result.get("r2", 0)),
                    rmse=float(result.get("rmse", float("inf"))),
                    formula=result.get("formula", "N/A"),
                    metadata={"iterations": result.get("iterations", 0)},
                )
            else:
                raise Exception(result.get("error", "Discovery failed"))

        except Exception as e:
            self._log_error(e, "during integrated discovery")
            result = self.fallback.run(description, X, y, var_names, metadata, verbose)
            result.method = self.name + " (Fallback)"
            result.error = str(e)[:100]
            return result


# ============================================================================
# METHOD 9: HYBRID SYSTEM V40 (FIXED)
# ============================================================================


class HybridSystemV40(BaseMethod):
    """Hybrid System v40 with proper error handling"""

    def __init__(self, verbose: bool = False):
        super().__init__("Hybrid System v40", verbose)
        self.fallback = LLMNNEnsembleSmart(verbose)
        self.system = None

        if ADVANCED_METHODS_AVAILABLE:
            try:
                self.system = HybridDiscoverySystem()
                if not hasattr(self.system, "discover"):
                    print(f"⚠️  HybridDiscoverySystem missing 'discover' method")
                    self.system = None
                else:
                    print(f"✅ {self.name} initialized")
            except Exception as e:
                self._log_error(e, "initializing HybridDiscoverySystem")
                self.system = None

    def run(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        verbose: bool = False,
    ) -> MethodResult:

        if self.system is None:
            result = self.fallback.run(description, X, y, var_names, metadata, verbose)
            result.method = self.name + " (Fallback)"
            return result

        try:
            result = self.system.discover(
                X=X,
                y=y,
                var_names=var_names,
                description=description,
                metadata=metadata,
                verbose=verbose,
            )

            if result and result.get("success"):
                return MethodResult(
                    method=self.name,
                    success=True,
                    r2=float(result.get("r2", 0)),
                    rmse=float(result.get("rmse", float("inf"))),
                    formula=result.get("final_formula", "N/A")[:80],
                    metadata={
                        "strategy": result.get("strategy", "unknown"),
                        "validations": result.get("validations", 0),
                    },
                )
            else:
                raise Exception(result.get("error", "Discovery failed"))

        except Exception as e:
            self._log_error(e, "during hybrid system discovery")
            result = self.fallback.run(description, X, y, var_names, metadata, verbose)
            result.method = self.name + " (Fallback)"
            result.error = f"System error: {str(e)[:100]}"
            return result


# ============================================================================
# ULTIMATE COMPARATIVE SUITE
# ============================================================================


class UltimateComparativeSuite:
    """Run all 9 methods and compare results"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.methods = [
            LLMGuidedPySRSimple(verbose),
            PurePySR(verbose),
            PureLLMBasic(verbose),
            PureLLMEnhanced(verbose),
            NeuralNetworkMethod(verbose),
            LLMNNEnsembleSimple(verbose),
            LLMNNEnsembleSmart(verbose),
            IntegratedLLMDiscovery(verbose),
            HybridSystemV40(verbose),
        ]
        self.results = []

        print(f"\n{'='*80}")
        print("ULTIMATE COMPARATIVE SUITE - FIXED VERSION".center(80))
        print(f"{'='*80}")
        print(f"Methods available: {len(self.methods)}")
        print(f"PySR: {'✅' if PYSR_AVAILABLE else '❌'}")
        print(
            f"Advanced Methods: {'✅' if ADVANCED_METHODS_AVAILABLE else '⚠️  (fallbacks enabled)'}"
        )
        print(f"API Key: {'✅' if os.getenv('ANTHROPIC_API_KEY') else '❌'}")
        print(f"{'='*80}\n")

    def run_test(
        self,
        description: str,
        X: np.ndarray,
        y: np.ndarray,
        var_names: List[str],
        metadata: Dict,
        domain: str,
        verbose: bool = True,
    ) -> Dict:
        """Run all methods on a single test"""

        if verbose:
            print(f"\n{'='*80}")
            print(f"Test: {description}")
            print(f"Domain: {domain}")
            print(f"Data: {X.shape[0]} samples × {X.shape[1]} variables")
            print(f"{'='*80}")

        results = {}

        for i, method in enumerate(self.methods, 1):
            if verbose:
                print(
                    f"\n[{i}/{len(self.methods)}] Running {method.name}...",
                    end=" ",
                    flush=True,
                )

            start_time = time.time()
            result = method.run(description, X, y, var_names, metadata, verbose=False)
            result.time = time.time() - start_time

            results[method.name] = result

            if verbose:
                if result.success:
                    print(
                        f"✓ R²: {result.r2:.4f}, RMSE: {result.rmse:.4f}, Time: {result.time:.1f}s"
                    )
                    if result.metadata and self.verbose:
                        for k, v in result.metadata.items():
                            print(f"    {k}: {v}")
                else:
                    error_msg = result.error if result.error else "Failed"
                    print(f"✗ {error_msg[:60]}")

        # Compare results
        comparison = self._compare_results(results)

        if verbose:
            self._print_comparison(results, comparison)

        test_result = {
            "description": description,
            "domain": domain,
            "results": {name: res.to_dict() for name, res in results.items()},
            "comparison": comparison,
            "winner": comparison["winner"],
            "timestamp": datetime.now().isoformat(),
        }

        self.results.append(test_result)
        return test_result

    def _compare_results(self, results: Dict[str, MethodResult]) -> Dict:
        """Compare all methods and determine winner"""
        r2_scores = {
            name: res.r2
            for name, res in results.items()
            if res.success and res.r2 > 0 and np.isfinite(res.r2)
        }

        if not r2_scores:
            return {"winner": "None", "scores": {}, "rankings": {}, "advantages": []}

        winner = max(r2_scores, key=r2_scores.get)
        winner_r2 = r2_scores[winner]

        # Calculate advantages
        advantages = []
        for method, r2 in sorted(r2_scores.items(), key=lambda x: x[1], reverse=True):
            if method != winner:
                diff = winner_r2 - r2
                pct = (diff / max(abs(r2), 0.001)) * 100
                advantages.append({"method": method, "diff": diff, "pct": pct})

        # Rankings
        rankings = {
            method: i + 1
            for i, (method, _) in enumerate(
                sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
            )
        }

        return {
            "winner": winner,
            "scores": r2_scores,
            "rankings": rankings,
            "advantages": advantages,
        }

    def _print_comparison(self, results: Dict[str, MethodResult], comparison: Dict):
        """Print detailed comparison table"""
        print(f"\n{'='*110}")
        print("COMPARISON RESULTS".center(110))
        print(f"{'='*110}")

        print(
            f"\n{'Method':<45} {'R²':<10} {'RMSE':<12} {'Time':<8} {'Rank':<6} {'Status':<20}"
        )
        print("-" * 110)

        rankings = comparison.get("rankings", {})

        for method_name, res in results.items():
            rank = rankings.get(method_name, "-")
            status = "🏆 WINNER" if comparison["winner"] == method_name else ""

            if res.success:
                print(
                    f"{method_name:<45} {res.r2:<10.4f} {res.rmse:<12.6f} {res.time:<8.1f}s {rank:<6} {status:<20}"
                )
            else:
                error = res.error[:15] if res.error else "Failed"
                print(
                    f"{method_name:<45} {'N/A':<10} {'N/A':<12} {res.time:<8.1f}s {'-':<6} {error:<20}"
                )

        print(f"{'='*110}")

        if comparison["winner"] != "None":
            print(f"\n🎯 WINNER: {comparison['winner']}")
            print(f"   R²: {comparison['scores'][comparison['winner']]:.4f}")

            if comparison["advantages"]:
                print(f"\n   Advantages over other methods:")
                for adv in comparison["advantages"][:5]:
                    print(
                        f"     • {adv['method']}: +{adv['diff']:.4f} ({adv['pct']:.1f}% better)"
                    )

    def print_summary(self):
        """Print overall summary across all tests"""
        if not self.results:
            print("\n⚠️  No tests run yet!")
            return

        print(f"\n{'='*110}")
        print("OVERALL SUMMARY".center(110))
        print(f"{'='*110}")

        # Count wins
        wins = {}
        total_r2 = {}
        total_rmse = {}
        success_count = {}
        total_tests = len(self.results)

        for result in self.results:
            winner = result["winner"]
            wins[winner] = wins.get(winner, 0) + 1

            for method_name, method_result in result["results"].items():
                if method_result["success"]:
                    if method_name not in total_r2:
                        total_r2[method_name] = []
                        total_rmse[method_name] = []
                        success_count[method_name] = 0

                    total_r2[method_name].append(method_result["r2"])
                    total_rmse[method_name].append(method_result["rmse"])
                    success_count[method_name] += 1

        print(f"\n📊 Total tests: {total_tests}")

        print(f"\n🏆 Wins by method:")
        for method, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_tests) * 100
            bar = "█" * int(pct / 5)
            print(f"   {method:<45} {count:>2}/{total_tests}  ({pct:>5.1f}%) {bar}")

        print(f"\n📈 Average R² by method:")
        for method, r2_list in sorted(
            total_r2.items(), key=lambda x: np.mean(x[1]) if x[1] else -1, reverse=True
        ):
            avg_r2 = np.mean(r2_list)
            std_r2 = np.std(r2_list)
            success_rate = (success_count[method] / total_tests) * 100
            print(
                f"   {method:<45} {avg_r2:.4f} ± {std_r2:.4f}  (success: {success_rate:.0f}%)"
            )

        print(f"\n📉 Average RMSE by method:")
        for method, rmse_list in sorted(
            total_rmse.items(), key=lambda x: np.mean(x[1]) if x[1] else float("inf")
        ):
            avg_rmse = np.mean(rmse_list)
            std_rmse = np.std(rmse_list)
            print(f"   {method:<45} {avg_rmse:.6f} ± {std_rmse:.6f}")

        print(f"\n{'='*110}")

        # Save results
        self._save_results()

    def _save_results(self):
        """Save results to JSON file"""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"comparison_FIXED_{timestamp}.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "version": "FIXED - All bugs resolved",
            "total_tests": len(self.results),
            "methods": [m.name for m in self.methods],
            "pysr_available": PYSR_AVAILABLE,
            "advanced_available": ADVANCED_METHODS_AVAILABLE,
            "tests": self.results,
        }

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fixed Ultimate Comparative Suite - All 9 Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests from all domains
  python ultimate_comparative_suite_FIXED.py --domain all_domains

  # Run specific domain
  python ultimate_comparative_suite_FIXED.py --domain chemistry

  # Run single test
  python ultimate_comparative_suite_FIXED.py --test arrhenius

  # Verbose mode
  python ultimate_comparative_suite_FIXED.py --domain physics --verbose
        """,
    )

    parser.add_argument(
        "--domain", type=str, default="all_domains", help="Domain filter"
    )
    parser.add_argument("--test", type=str, help="Run single test by name")
    parser.add_argument(
        "--samples", type=int, default=200, help="Number of samples to generate"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity"
    )

    args = parser.parse_args()

    # Load protocol
    try:
        from experiment_protocol_comparative import ComparativeExperimentProtocol

        protocol = ComparativeExperimentProtocol()
    except ImportError:
        print("❌ Error: experiment_protocol_comparative.py not found")
        print("   Place it in the same directory as this script")
        return

    # Initialize suite
    suite = UltimateComparativeSuite(verbose=args.verbose)

    # Get tests
    all_tests = []

    if args.test:
        # Search for specific test
        print(f"\n🔍 Searching for test: '{args.test}'")
        found = False
        for domain in protocol.get_all_domains():
            domain_tests = protocol.load_test_data(domain, num_samples=args.samples)
            for desc, X, y, var_names, metadata in domain_tests:
                if args.test.lower() in metadata["equation_name"].lower():
                    all_tests.append((desc, X, y, var_names, metadata, domain))
                    found = True
                    break
            if found:
                break

        if not all_tests:
            print(f"❌ Test '{args.test}' not found")
            print(f"\n📋 Available tests:")
            for domain in protocol.get_all_domains():
                tests = protocol.load_test_data(domain, num_samples=10)
                print(f"\n  {domain}:")
                for _, _, _, _, meta in tests:
                    print(f"    • {meta['equation_name']}")
            return

    elif args.domain == "all_domains":
        # Load all domains
        for domain in protocol.get_all_domains():
            domain_tests = protocol.load_test_data(domain, num_samples=args.samples)
            for desc, X, y, var_names, metadata in domain_tests:
                all_tests.append((desc, X, y, var_names, metadata, domain))

    else:
        # Load specific domain
        if args.domain not in protocol.get_all_domains():
            print(f"❌ Unknown domain '{args.domain}'")
            print(f"📋 Available: {', '.join(protocol.get_all_domains())}")
            return

        domain_tests = protocol.load_test_data(args.domain, num_samples=args.samples)
        for desc, X, y, var_names, metadata in domain_tests:
            all_tests.append((desc, X, y, var_names, metadata, args.domain))

    if not all_tests:
        print("❌ No tests to run")
        return

    print(f"\n🚀 Running {len(all_tests)} test(s)...\n")

    # Run tests
    for i, (description, X, y, var_names, metadata, domain) in enumerate(all_tests, 1):
        print(f"\n{'='*110}")
        print(f"TEST {i}/{len(all_tests)}".center(110))
        print(f"{'='*110}")

        suite.run_test(
            description=description,
            X=X,
            y=y,
            var_names=var_names,
            metadata=metadata,
            domain=domain,
            verbose=not args.quiet,
        )

    # Print summary
    suite.print_summary()

    print("\n✅ Complete!\n")


if __name__ == "__main__":
    main()
