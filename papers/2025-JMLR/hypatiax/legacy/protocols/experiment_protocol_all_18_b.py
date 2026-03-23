"""
Experiment Protocol B v2.0: 18 Multi-Domain Test Cases - OPTIMIZED
====================================================================
Focus: Physics, Chemistry, Biology, Engineering, Mathematics, Economics

UPDATES IN v2.0:
- ✅ Aligned with 8_new_all.py v3.8 structure
- ✅ Enhanced metadata with difficulty levels
- ✅ Better data generation with controlled noise
- ✅ Comprehensive unit specifications
- ✅ Ground truth expressions for validation
- ✅ Variable role specifications (constant vs varying)
- ✅ Compatible with suite v4.0 optimized

Author: HypatiaX Team
Version: 2.0 Optimized
Date: 2026-01-04
"""

import numpy as np
from typing import List, Tuple, Dict
import json
import os


class ExperimentProtocolB:
    """18 test cases across 6 diverse scientific domains - OPTIMIZED v2.0"""

    @staticmethod
    def get_all_domains() -> List[str]:
        """Return list of all experimental domains."""
        return [
            "physics",
            "chemistry",
            "biology",
            "engineering",
            "mathematics",
            "economics",
        ]

    @staticmethod
    def load_test_data(
        domain: str, num_samples: int = 300
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """
        Load test data for Protocol B (18 multi-domain cases).

        Args:
            domain: Domain name (physics, chemistry, biology, engineering, mathematics, economics)
            num_samples: Number of samples to generate

        Returns:
            List of (description, X, y, variable_names, metadata) tuples
        """
        np.random.seed(42)
        test_cases = []

        if domain == "physics":
            # ========== PHYSICS (4 tests) ==========

            # 1. Kinetic Energy - ALIGNED with 8_new_all.py
            m = np.random.uniform(1, 100, num_samples)
            v = np.random.uniform(0, 50, num_samples)
            X = np.column_stack([m, v])
            y_true = 0.5 * m * v**2
            y = y_true + np.random.normal(
                0, np.abs(y_true) * 0.01, num_samples
            )  # 1% noise
            test_cases.append(
                (
                    "Kinetic Energy: KE = 0.5*m*v²",
                    X,
                    y,
                    ["m", "v"],
                    {
                        "equation_name": "kinetic_energy",
                        "difficulty": "easy",
                        "formula_type": "power_law",
                        "ground_truth": "0.5 * m * v**2",
                        "protocol": "Protocol_B",
                        "units": {"m": "kg", "v": "m/s", "KE": "J"},
                        "variable_roles": {"m": "varying", "v": "varying"},
                        "expected_patterns": ["multiplicative", "power_law"],
                        "noise_level": 0.01,
                    },
                )
            )

            # 2. Ohm's Law - ALIGNED with 8_new_all.py
            I = np.random.uniform(0.1, 10, num_samples)
            R = np.random.uniform(1, 1000, num_samples)
            X = np.column_stack([I, R])
            y_true = I * R
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.01, num_samples)
            test_cases.append(
                (
                    "Ohm's Law: V = I*R",
                    X,
                    y,
                    ["I", "R"],
                    {
                        "equation_name": "ohms_law",
                        "difficulty": "easy",
                        "formula_type": "linear_multiplicative",
                        "ground_truth": "I * R",
                        "protocol": "Protocol_B",
                        "units": {"I": "A", "R": "Ω", "V": "V"},
                        "variable_roles": {"I": "varying", "R": "varying"},
                        "expected_patterns": ["multiplicative"],
                        "noise_level": 0.01,
                    },
                )
            )

            # 3. Ideal Gas Law
            n = np.random.uniform(0.1, 10, num_samples)
            R = np.full(num_samples, 8.314)
            T = np.random.uniform(200, 400, num_samples)
            V = np.random.uniform(0.001, 0.1, num_samples)
            X = np.column_stack([n, R, T, V])
            y_true = n * R * T / V
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.01, num_samples)
            test_cases.append(
                (
                    "Ideal Gas Law: P = nRT/V",
                    X,
                    y,
                    ["n", "R", "T", "V"],
                    {
                        "equation_name": "ideal_gas_law",
                        "difficulty": "medium",
                        "formula_type": "rational",
                        "ground_truth": "n * R * T / V",
                        "protocol": "Protocol_B",
                        "units": {
                            "n": "mol",
                            "R": "J/(mol*K)",
                            "T": "K",
                            "V": "m³",
                            "P": "Pa",
                        },
                        "variable_roles": {
                            "n": "varying",
                            "R": "constant",
                            "T": "varying",
                            "V": "varying",
                        },
                        "expected_patterns": ["multiplicative", "division"],
                        "noise_level": 0.01,
                    },
                )
            )

            # 4. Projectile Motion - Range
            v = np.random.uniform(10, 50, num_samples)
            theta = np.random.uniform(0.1, 1.4, num_samples)  # radians
            g = np.full(num_samples, 9.81)
            X = np.column_stack([v, theta, g])
            y_true = (v**2 * np.sin(2 * theta)) / g
            y = y_true + np.random.normal(
                0, np.abs(y_true) * 0.02, num_samples
            )  # 2% noise
            test_cases.append(
                (
                    "Projectile Motion: Range = (v²*sin(2θ))/g",
                    X,
                    y,
                    ["v", "theta", "g"],
                    {
                        "equation_name": "projectile_motion",
                        "difficulty": "hard",
                        "formula_type": "trigonometric",
                        "ground_truth": "(v**2 * np.sin(2 * theta)) / g",
                        "protocol": "Protocol_B",
                        "units": {"v": "m/s", "theta": "rad", "g": "m/s²", "R": "m"},
                        "variable_roles": {
                            "v": "varying",
                            "theta": "varying",
                            "g": "constant",
                        },
                        "expected_patterns": ["trigonometric", "power_law"],
                        "noise_level": 0.02,
                    },
                )
            )

        elif domain == "chemistry":
            # ========== CHEMISTRY (3 tests) ==========

            # 1. Arrhenius Equation
            A = np.full(num_samples, 1e11)
            Ea = np.full(num_samples, 80000)
            R = np.full(num_samples, 8.314)
            T = np.random.uniform(273, 373, num_samples)
            X = np.column_stack([A, Ea, R, T])
            y_true = A * np.exp(-Ea / (R * T))
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.01, num_samples)
            test_cases.append(
                (
                    "Arrhenius Equation: k = A*exp(-Ea/(R*T))",
                    X,
                    y,
                    ["A", "Ea", "R", "T"],
                    {
                        "equation_name": "arrhenius_equation",
                        "difficulty": "hard",
                        "formula_type": "exponential",
                        "ground_truth": "A * np.exp(-Ea / (R * T))",
                        "protocol": "Protocol_B",
                        "units": {
                            "A": "1/s",
                            "Ea": "J/mol",
                            "R": "J/(mol*K)",
                            "T": "K",
                            "k": "1/s",
                        },
                        "variable_roles": {
                            "A": "constant",
                            "Ea": "constant",
                            "R": "constant",
                            "T": "varying",
                        },
                        "expected_patterns": ["exponential", "division"],
                        "noise_level": 0.01,
                    },
                )
            )

            # 2. Henderson-Hasselbalch Equation - ALIGNED with 8_new_all.py
            pKa = np.full(num_samples, 6.5)
            A_minus = np.random.uniform(0.01, 1.0, num_samples)
            HA = np.random.uniform(0.01, 1.0, num_samples)
            X = np.column_stack([pKa, A_minus, HA])
            y_true = pKa + np.log10(A_minus / (HA + 1e-12))
            y = y_true + np.random.normal(0, 0.05, num_samples)  # Small absolute noise
            test_cases.append(
                (
                    "Henderson-Hasselbalch: pH = pKa + log10([A-]/[HA])",
                    X,
                    y,
                    ["pKa", "A_minus", "HA"],
                    {
                        "equation_name": "henderson_hasselbalch",
                        "difficulty": "medium",
                        "formula_type": "logarithmic",
                        "ground_truth": "pKa + np.log10(A_minus / HA)",
                        "protocol": "Protocol_B",
                        "units": {
                            "pKa": "dimensionless",
                            "A_minus": "mol/L",
                            "HA": "mol/L",
                            "pH": "dimensionless",
                        },
                        "variable_roles": {
                            "pKa": "constant",
                            "A_minus": "varying",
                            "HA": "varying",
                        },
                        "expected_patterns": ["logarithmic", "additive"],
                        "noise_level": 0.05,
                    },
                )
            )

            # 3. Nernst Equation - ALIGNED with 8_new_all.py
            E0 = np.random.uniform(0.1, 1.5, num_samples)
            R = np.full(num_samples, 8.314)
            T = np.random.uniform(273, 373, num_samples)
            n = np.random.randint(1, 3, num_samples).astype(float)
            F = np.full(num_samples, 96485)
            Q = np.random.uniform(0.01, 100, num_samples)
            X = np.column_stack([E0, R, T, n, F, Q])
            y_true = E0 - (R * T / (n * F)) * np.log(Q)
            y = y_true + np.random.normal(0, 0.01, num_samples)
            test_cases.append(
                (
                    "Nernst Equation: E = E0 - (RT/nF)*ln(Q)",
                    X,
                    y,
                    ["E0", "R", "T", "n", "F", "Q"],
                    {
                        "equation_name": "nernst_equation",
                        "difficulty": "hard",
                        "formula_type": "logarithmic_complex",
                        "ground_truth": "E0 - (R * T / (n * F)) * np.log(Q)",
                        "protocol": "Protocol_B",
                        "units": {
                            "E0": "V",
                            "R": "J/(mol*K)",
                            "T": "K",
                            "n": "dimensionless",
                            "F": "C/mol",
                            "Q": "dimensionless",
                            "E": "V",
                        },
                        "variable_roles": {
                            "E0": "varying",
                            "R": "constant",
                            "T": "varying",
                            "n": "constant",
                            "F": "constant",
                            "Q": "varying",
                        },
                        "expected_patterns": ["logarithmic", "subtraction", "division"],
                        "noise_level": 0.01,
                    },
                )
            )

        elif domain == "biology":
            # ========== BIOLOGY (3 tests) ==========

            # 1. Michaelis-Menten Kinetics
            Vmax = np.full(num_samples, 50.0)
            S = np.random.uniform(0.1, 50, num_samples)
            Km = np.full(num_samples, 10.0)
            X = np.column_stack([Vmax, S, Km])
            y_true = (Vmax * S) / (Km + S)
            y = y_true + np.random.normal(0, 0.5, num_samples)
            test_cases.append(
                (
                    "Michaelis-Menten: v = (Vmax*[S])/(Km+[S])",
                    X,
                    y,
                    ["Vmax", "S", "Km"],
                    {
                        "equation_name": "michaelis_menten",
                        "difficulty": "medium",
                        "formula_type": "rational",
                        "ground_truth": "(Vmax * S) / (Km + S)",
                        "protocol": "Protocol_B",
                        "units": {
                            "Vmax": "mol/(L*s)",
                            "S": "mol/L",
                            "Km": "mol/L",
                            "v": "mol/(L*s)",
                        },
                        "variable_roles": {
                            "Vmax": "constant",
                            "S": "varying",
                            "Km": "constant",
                        },
                        "expected_patterns": ["rational", "saturation"],
                        "noise_level": 0.5,
                    },
                )
            )

            # 2. Logistic Growth Model - ALIGNED with 8_new_all.py
            r = np.random.uniform(0.1, 0.5, num_samples)
            N = np.random.uniform(10, 900, num_samples)
            K = np.random.uniform(1000, 2000, num_samples)
            X = np.column_stack([r, N, K])
            y_true = r * N * (1 - N / K)
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.02, num_samples)
            test_cases.append(
                (
                    "Logistic Growth: dN/dt = r*N*(1-N/K)",
                    X,
                    y,
                    ["r", "N", "K"],
                    {
                        "equation_name": "logistic_growth",
                        "difficulty": "medium",
                        "formula_type": "nonlinear",
                        "ground_truth": "r * N * (1 - N / K)",
                        "protocol": "Protocol_B",
                        "units": {
                            "r": "1/s",
                            "N": "dimensionless",
                            "K": "dimensionless",
                            "dNdt": "1/s",
                        },
                        "variable_roles": {
                            "r": "constant",
                            "N": "varying",
                            "K": "constant",
                        },
                        "expected_patterns": ["multiplicative", "nonlinear"],
                        "noise_level": 0.02,
                    },
                )
            )

            # 3. Allometric Scaling Law
            a = np.full(num_samples, 3.5)
            M = np.random.uniform(0.1, 100, num_samples)
            b = np.full(num_samples, 0.75)
            X = np.column_stack([a, M, b])
            y_true = a * M**b
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.02, num_samples)
            test_cases.append(
                (
                    "Allometric Scaling: Y = a*M^b",
                    X,
                    y,
                    ["a", "M", "b"],
                    {
                        "equation_name": "allometric_scaling",
                        "difficulty": "easy",
                        "formula_type": "power_law",
                        "ground_truth": "a * M**b",
                        "protocol": "Protocol_B",
                        "units": {
                            "a": "W/kg^0.75",
                            "M": "kg",
                            "b": "dimensionless",
                            "Y": "W",
                        },
                        "variable_roles": {
                            "a": "constant",
                            "M": "varying",
                            "b": "constant",
                        },
                        "expected_patterns": ["power_law", "multiplicative"],
                        "noise_level": 0.02,
                    },
                )
            )

        elif domain == "engineering":
            # ========== ENGINEERING (3 tests) ==========

            # 1. Reynolds Number
            rho = np.random.uniform(800, 1200, num_samples)
            v = np.random.uniform(0.1, 10, num_samples)
            L = np.random.uniform(0.01, 1, num_samples)
            mu = np.random.uniform(0.001, 0.01, num_samples)
            X = np.column_stack([rho, v, L, mu])
            y_true = (rho * v * L) / mu
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.01, num_samples)
            test_cases.append(
                (
                    "Reynolds Number: Re = (ρ*v*L)/μ",
                    X,
                    y,
                    ["rho", "v", "L", "mu"],
                    {
                        "equation_name": "reynolds_number",
                        "difficulty": "easy",
                        "formula_type": "rational",
                        "ground_truth": "(rho * v * L) / mu",
                        "protocol": "Protocol_B",
                        "units": {
                            "rho": "kg/m³",
                            "v": "m/s",
                            "L": "m",
                            "mu": "Pa·s",
                            "Re": "dimensionless",
                        },
                        "variable_roles": {
                            "rho": "varying",
                            "v": "varying",
                            "L": "varying",
                            "mu": "varying",
                        },
                        "expected_patterns": ["multiplicative", "division"],
                        "noise_level": 0.01,
                    },
                )
            )

            # 2. Hooke's Law
            k = np.random.uniform(10, 1000, num_samples)
            x = np.random.uniform(-0.5, 0.5, num_samples)
            X = np.column_stack([k, x])
            y_true = k * np.abs(x)
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.01, num_samples)
            test_cases.append(
                (
                    "Hooke's Law: F = k*|x|",
                    X,
                    y,
                    ["k", "x"],
                    {
                        "equation_name": "hookes_law",
                        "difficulty": "easy",
                        "formula_type": "linear",
                        "ground_truth": "k * np.abs(x)",
                        "protocol": "Protocol_B",
                        "units": {"k": "N/m", "x": "m", "F": "N"},
                        "variable_roles": {"k": "varying", "x": "varying"},
                        "expected_patterns": ["multiplicative", "linear"],
                        "noise_level": 0.01,
                    },
                )
            )

            # 3. Bernoulli's Equation - ALIGNED with 8_new_all.py (ENHANCED)
            P = np.random.uniform(1e5, 2e5, num_samples)
            rho = np.full(num_samples, 1000)
            v = np.random.uniform(0, 15, num_samples)
            g = np.full(num_samples, 9.81)
            h = np.random.uniform(0, 10, num_samples)
            X = np.column_stack([P, rho, v, g, h])
            y_true = P + 0.5 * rho * v**2 + rho * g * h
            y = y_true + np.random.normal(
                0, np.abs(y_true) * 0.005, num_samples
            )  # 0.5% noise
            test_cases.append(
                (
                    "Bernoulli's Equation: P + 0.5*ρ*v² + ρ*g*h",
                    X,
                    y,
                    ["P", "rho", "v", "g", "h"],
                    {
                        "equation_name": "bernoulli_equation",
                        "difficulty": "hard",
                        "formula_type": "additive_polynomial",
                        "ground_truth": "P + 0.5 * rho * v**2 + rho * g * h",
                        "protocol": "Protocol_B",
                        "units": {
                            "P": "Pa",
                            "rho": "kg/m³",
                            "v": "m/s",
                            "g": "m/s²",
                            "h": "m",
                            "E": "Pa",
                        },
                        "variable_roles": {
                            "P": "varying",
                            "rho": "constant",
                            "v": "varying",
                            "g": "constant",
                            "h": "varying",
                        },
                        "expected_patterns": [
                            "additive",
                            "power_law",
                            "multiplicative",
                        ],
                        "noise_level": 0.005,
                        "use_enhanced_config": True,  # Flag for special handling
                    },
                )
            )

        elif domain == "mathematics":
            # ========== MATHEMATICS (3 tests) ==========

            # 1. Pythagorean Theorem
            a = np.random.uniform(1, 10, num_samples)
            b = np.random.uniform(1, 10, num_samples)
            X = np.column_stack([a, b])
            y_true = np.sqrt(a**2 + b**2)
            y = y_true + np.random.normal(0, 0.01, num_samples)
            test_cases.append(
                (
                    "Pythagorean Theorem: c = sqrt(a² + b²)",
                    X,
                    y,
                    ["a", "b"],
                    {
                        "equation_name": "pythagorean_theorem",
                        "difficulty": "easy",
                        "formula_type": "power_law",
                        "ground_truth": "np.sqrt(a**2 + b**2)",
                        "protocol": "Protocol_B",
                        "units": {"a": "m", "b": "m", "c": "m"},
                        "variable_roles": {"a": "varying", "b": "varying"},
                        "expected_patterns": ["power_law", "sqrt"],
                        "noise_level": 0.01,
                    },
                )
            )

            # 2. Compound Interest
            P = np.random.uniform(1000, 10000, num_samples)
            r = np.random.uniform(0.01, 0.1, num_samples)
            n = np.random.choice([1, 4, 12], num_samples).astype(float)
            t = np.random.uniform(1, 20, num_samples)
            X = np.column_stack([P, r, n, t])
            y_true = P * (1 + r / n) ** (n * t)
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.01, num_samples)
            test_cases.append(
                (
                    "Compound Interest: A = P*(1+r/n)^(n*t)",
                    X,
                    y,
                    ["P", "r", "n", "t"],
                    {
                        "equation_name": "compound_interest",
                        "difficulty": "medium",
                        "formula_type": "exponential",
                        "ground_truth": "P * (1 + r/n)**(n*t)",
                        "protocol": "Protocol_B",
                        "units": {
                            "P": "USD",
                            "r": "1/year",
                            "n": "1/year",
                            "t": "year",
                            "A": "USD",
                        },
                        "variable_roles": {
                            "P": "varying",
                            "r": "varying",
                            "n": "varying",
                            "t": "varying",
                        },
                        "expected_patterns": ["exponential", "multiplicative"],
                        "noise_level": 0.01,
                    },
                )
            )

            # 3. Quadratic Formula (Discriminant)
            a = np.random.uniform(-5, 5, num_samples)
            a[np.abs(a) < 0.1] = 1.0  # Avoid near-zero values
            b = np.random.uniform(-10, 10, num_samples)
            c = np.random.uniform(-5, 5, num_samples)
            X = np.column_stack([a, b, c])
            y_true = b**2 - 4 * a * c
            y = y_true + np.random.normal(0, 1.0, num_samples)
            test_cases.append(
                (
                    "Quadratic Discriminant: Δ = b² - 4ac",
                    X,
                    y,
                    ["a", "b", "c"],
                    {
                        "equation_name": "quadratic_discriminant",
                        "difficulty": "easy",
                        "formula_type": "polynomial",
                        "ground_truth": "b**2 - 4*a*c",
                        "protocol": "Protocol_B",
                        "units": {
                            "a": "dimensionless",
                            "b": "dimensionless",
                            "c": "dimensionless",
                            "delta": "dimensionless",
                        },
                        "variable_roles": {
                            "a": "varying",
                            "b": "varying",
                            "c": "varying",
                        },
                        "expected_patterns": ["polynomial", "subtraction"],
                        "noise_level": 1.0,
                    },
                )
            )

        elif domain == "economics":
            # ========== ECONOMICS (2 tests) ==========

            # 1. Price Elasticity of Demand - ALIGNED with 8_new_all.py
            Q = np.random.uniform(100, 1000, num_samples)
            delta_Q = np.random.uniform(-50, 50, num_samples)
            P = np.random.uniform(10, 100, num_samples)
            delta_P = np.random.uniform(-5, 5, num_samples)
            # Avoid division by zero
            delta_P[np.abs(delta_P) < 0.1] = 0.1
            X = np.column_stack([Q, delta_Q, P, delta_P])
            y_true = (delta_Q / (Q + 1e-10)) / ((delta_P / (P + 1e-10)) + 1e-10)
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.05, num_samples)
            test_cases.append(
                (
                    "Price Elasticity: Ed = (ΔQ/Q)/(ΔP/P)",
                    X,
                    y,
                    ["Q", "delta_Q", "P", "delta_P"],
                    {
                        "equation_name": "elasticity_demand",
                        "difficulty": "medium",
                        "formula_type": "rational",
                        "ground_truth": "(delta_Q / Q) / (delta_P / P)",
                        "protocol": "Protocol_B",
                        "units": {
                            "Q": "dimensionless",
                            "delta_Q": "dimensionless",
                            "P": "dimensionless",
                            "delta_P": "dimensionless",
                            "Ed": "dimensionless",
                        },
                        "variable_roles": {
                            "Q": "varying",
                            "delta_Q": "varying",
                            "P": "varying",
                            "delta_P": "varying",
                        },
                        "expected_patterns": ["rational", "division"],
                        "noise_level": 0.05,
                    },
                )
            )

            # 2. Cobb-Douglas Production Function - ALIGNED with 8_new_all.py
            A = np.random.uniform(1, 5, num_samples)
            K = np.random.uniform(100, 1000, num_samples)
            L = np.random.uniform(10, 100, num_samples)
            alpha = np.full(num_samples, 0.3)
            beta = np.full(num_samples, 0.7)
            X = np.column_stack([A, K, L, alpha, beta])
            y_true = A * K**alpha * L**beta
            y = y_true + np.random.normal(0, np.abs(y_true) * 0.02, num_samples)
            test_cases.append(
                (
                    "Cobb-Douglas: Y = A*K^α*L^β",
                    X,
                    y,
                    ["A", "K", "L", "alpha", "beta"],
                    {
                        "equation_name": "cobb_douglas",
                        "difficulty": "medium",
                        "formula_type": "power_law",
                        "ground_truth": "A * K**alpha * L**beta",
                        "protocol": "Protocol_B",
                        "units": {
                            "A": "dimensionless",
                            "K": "dimensionless",
                            "L": "dimensionless",
                            "alpha": "dimensionless",
                            "beta": "dimensionless",
                            "Y": "dimensionless",
                        },
                        "variable_roles": {
                            "A": "varying",
                            "K": "varying",
                            "L": "varying",
                            "alpha": "constant",
                            "beta": "constant",
                        },
                        "expected_patterns": ["power_law", "multiplicative"],
                        "noise_level": 0.02,
                    },
                )
            )

        return test_cases

    @staticmethod
    def get_domain_description(domain: str) -> str:
        """Get domain description."""
        descriptions = {
            "physics": "Physics - mechanics, thermodynamics, electromagnetism",
            "chemistry": "Chemistry - kinetics, equilibrium, electrochemistry",
            "biology": "Biology - enzyme kinetics, population dynamics, allometry",
            "engineering": "Engineering - fluid dynamics, mechanics, design",
            "mathematics": "Mathematics - geometry, finance, algebra",
            "economics": "Economics - elasticity, production functions",
        }
        return descriptions.get(domain, "Unknown domain")

    @staticmethod
    def get_protocol_statistics() -> Dict:
        """Get comprehensive protocol statistics."""
        stats = {
            "version": "2.0",
            "total_tests": 18,
            "domains": {
                "physics": 4,
                "chemistry": 3,
                "biology": 3,
                "engineering": 3,
                "mathematics": 3,
                "economics": 2,
            },
            "difficulty_distribution": {"easy": 6, "medium": 8, "hard": 4},
            "formula_types": {
                "power_law": 5,
                "rational": 4,
                "exponential": 2,
                "logarithmic": 3,
                "trigonometric": 1,
                "polynomial": 2,
                "linear": 1,
            },
        }
        return stats

    @staticmethod
    def save_protocol_documentation(
        filepath: str = "docs/experiment_protocol_b_v2.json",
    ):
        """Save protocol documentation with enhanced metadata."""
        protocol_doc = {
            "title": "Experiment Protocol B: 18 Multi-Domain Cases",
            "version": "2.0 Optimized",
            "date": "2026-01-04",
            "author": "HypatiaX Team",
            "total_tests": 18,
            "alignment": "8_new_all.py v3.8 compatible",
            "domains": {},
        }

        for domain in ExperimentProtocolB.get_all_domains():
            test_cases = ExperimentProtocolB.load_test_data(domain, num_samples=10)
            protocol_doc["domains"][domain] = {
                "description": ExperimentProtocolB.get_domain_description(domain),
                "num_test_cases": len(test_cases),
                "test_cases": [
                    {
                        "description": desc,
                        "variables": vars,
                        "equation_name": meta.get("equation_name"),
                        "difficulty": meta["difficulty"],
                        "formula_type": meta["formula_type"],
                        "ground_truth": meta["ground_truth"],
                        "units": meta.get("units", {}),
                        "variable_roles": meta.get("variable_roles", {}),
                        "expected_patterns": meta.get("expected_patterns", []),
                        "noise_level": meta.get("noise_level", 0.01),
                    }
                    for desc, _, _, vars, meta in test_cases
                ],
            }

        # Add protocol statistics
        protocol_doc["statistics"] = ExperimentProtocolB.get_protocol_statistics()

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(protocol_doc, f, indent=2)

        print(f"✅ Protocol B v2.0 documentation saved to: {filepath}")
        return protocol_doc


if __name__ == "__main__":
    protocol = ExperimentProtocolB()

    print("=" * 80)
    print(
        "EXPERIMENT PROTOCOL B v2.0: 18 MULTI-DOMAIN TEST CASES (OPTIMIZED)".center(80)
    )
    print("=" * 80)
    print(f"Version: 2.0 | Date: 2026-01-04 | Alignment: 8_new_all.py v3.8")
    print("=" * 80)

    total_tests = 0
    difficulty_count = {"easy": 0, "medium": 0, "hard": 0}

    for domain in protocol.get_all_domains():
        test_cases = protocol.load_test_data(domain, num_samples=10)
        total_tests += len(test_cases)

        print(f"\n{'=' * 80}")
        print(f"{domain.upper()} - {len(test_cases)} tests")
        print(f"Description: {protocol.get_domain_description(domain)}")
        print(f"{'=' * 80}")

        for i, (desc, X, y, vars, meta) in enumerate(test_cases, 1):
            difficulty_count[meta["difficulty"]] += 1
            print(f"\n  {i}. {desc}")
            print(f"     Equation: {meta['equation_name']}")
            print(f"     Variables: {', '.join(vars)}")
            print(f"     Ground truth: {meta['ground_truth']}")
            print(
                f"     Difficulty: {meta['difficulty']} | Type: {meta['formula_type']}"
            )
            print(f"     Data shape: X{X.shape}, y{y.shape}")
            print(f"     Noise level: {meta.get('noise_level', 'N/A')}")
            if meta.get("variable_roles"):
                const_vars = [
                    k for k, v in meta["variable_roles"].items() if v == "constant"
                ]
                vary_vars = [
                    k for k, v in meta["variable_roles"].items() if v == "varying"
                ]
                if const_vars:
                    print(f"     Constants: {', '.join(const_vars)}")
                if vary_vars:
                    print(f"     Varying: {', '.join(vary_vars)}")

    print(f"\n{'=' * 80}")
    print(f"PROTOCOL SUMMARY".center(80))
    print(f"{'=' * 80}")
    print(f"Total test cases: {total_tests}")
    print(f"Domains covered: {len(protocol.get_all_domains())}")
    print(f"\nDifficulty distribution:")
    for diff, count in difficulty_count.items():
        print(f"  - {diff.capitalize()}: {count} tests")

    stats = protocol.get_protocol_statistics()
    print(f"\nFormula types:")
    for ftype, count in stats["formula_types"].items():
        print(f"  - {ftype}: {count} tests")

    print(f"\n{'=' * 80}")
    print(f"✅ Protocol B v2.0 ready for testing")
    print(f"{'=' * 80}")

    # Save documentation
    protocol.save_protocol_documentation()
