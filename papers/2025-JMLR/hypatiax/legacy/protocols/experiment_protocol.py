"""
Research-Aligned 5-Domain Experiment Protocol for Pure LLM Formula Discovery
============================================================================

Domains matching the JMLR paper research focus:
1. Materials Science (Hall-Petch, strength-grain size)
2. Fluid Dynamics (Darcy-Weisbach, friction factor)
3. Thermodynamics (heat transfer, efficiency)
4. Mechanics (stress-strain, buckling)
5. Chemistry (reaction kinetics, equilibrium)
"""

import numpy as np
from typing import List, Tuple, Dict
import json


class ExperimentProtocol:
    """
    Experimental protocol for evaluating pure LLM formula discovery across 5 scientific domains.
    """

    @staticmethod
    def get_all_domains() -> List[str]:
        """Return list of all experimental domains."""
        return ["materials", "fluids", "thermodynamics", "mechanics", "chemistry"]

    @staticmethod
    def load_test_data(
        domain: str, num_samples: int = 100
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List[str], Dict]]:
        """
        Load test data for evaluation across 5 scientific/engineering domains.

        Args:
            domain: Domain to load data for
            num_samples: Number of samples to generate

        Returns:
            List of (description, X, y, variable_names, metadata) tuples
        """
        np.random.seed(42)  # For reproducibility
        test_cases = []

        if domain == "materials":
            # ==================== MATERIALS SCIENCE DOMAIN ====================

            # 1. Hall-Petch Relationship (CRITICAL TEST CASE)
            grain_size = np.random.uniform(0.5, 50.0, num_samples)  # micrometers
            sigma_0 = 50  # MPa (base yield strength)
            k = 15  # MPa·μm^0.5 (Hall-Petch constant)
            X = grain_size.reshape(-1, 1)
            y = sigma_0 + k / np.sqrt(grain_size)
            test_cases.append(
                (
                    "Hall-Petch relationship: yield strength as function of grain size",
                    X,
                    y,
                    ["grain_size"],
                    {
                        "difficulty": "medium",
                        "formula_type": "power_law",
                        "ground_truth": "σ_y = σ_0 + k/√d",
                        "domain_specific": True,
                        "constants": {"sigma_0": 50, "k": 15},
                        "units": {"grain_size": "μm", "yield_strength": "MPa"},
                        "extrapolation_test": True,  # Key for testing extrapolation
                    },
                )
            )

            # 2. Young's Modulus - Porosity (Gibson-Ashby)
            porosity = np.random.uniform(0.1, 0.7, num_samples)
            E_0 = 200  # GPa (solid material)
            X = porosity.reshape(-1, 1)
            y = E_0 * (1 - porosity) ** 2
            test_cases.append(
                (
                    "Gibson-Ashby: Young's modulus as function of porosity",
                    X,
                    y,
                    ["porosity"],
                    {
                        "difficulty": "medium",
                        "formula_type": "power_law",
                        "ground_truth": "E = E_0(1-φ)²",
                        "domain_specific": True,
                        "constants": {"E_0": 200},
                        "units": {"porosity": "fraction", "modulus": "GPa"},
                    },
                )
            )

            # 3. Thermal Expansion
            temp_change = np.random.uniform(0, 200, num_samples)  # Celsius
            alpha = 12e-6  # 1/°C (coefficient of thermal expansion)
            L_0 = 1000  # mm (original length)
            X = temp_change.reshape(-1, 1)
            y = L_0 * (1 + alpha * temp_change)
            test_cases.append(
                (
                    "Linear thermal expansion of materials",
                    X,
                    y,
                    ["temperature_change"],
                    {
                        "difficulty": "easy",
                        "formula_type": "linear",
                        "ground_truth": "L = L_0(1 + α·ΔT)",
                        "domain_specific": False,
                        "constants": {"L_0": 1000, "alpha": 12e-6},
                        "units": {"temp": "°C", "length": "mm"},
                    },
                )
            )

            # 4. Creep Strain Rate (Power Law)
            stress = np.random.uniform(10, 100, num_samples)  # MPa
            A = 1e-10  # material constant
            n = 3  # stress exponent
            X = stress.reshape(-1, 1)
            y = A * stress**n
            test_cases.append(
                (
                    "Power law creep: strain rate as function of applied stress",
                    X,
                    y,
                    ["stress"],
                    {
                        "difficulty": "easy",
                        "formula_type": "power_law",
                        "ground_truth": "ε̇ = A·σⁿ",
                        "domain_specific": True,
                        "constants": {"A": 1e-10, "n": 3},
                        "units": {"stress": "MPa", "strain_rate": "1/s"},
                    },
                )
            )

        elif domain == "fluids":
            # ==================== FLUID DYNAMICS DOMAIN ====================

            # 1. Darcy-Weisbach Equation (CRITICAL TEST CASE)
            friction_factor = np.random.uniform(0.01, 0.08, num_samples)
            length = np.random.uniform(10, 100, num_samples)  # meters
            velocity = np.random.uniform(0.5, 5.0, num_samples)  # m/s
            diameter = np.random.uniform(0.05, 0.5, num_samples)  # meters
            g = 9.81  # m/s²
            X = np.column_stack([friction_factor, length, velocity, diameter])
            y = friction_factor * (length / diameter) * (velocity**2) / (2 * g)
            test_cases.append(
                (
                    "Darcy-Weisbach equation: pressure head loss in pipe flow",
                    X,
                    y,
                    [
                        "friction_factor",
                        "pipe_length",
                        "flow_velocity",
                        "pipe_diameter",
                    ],
                    {
                        "difficulty": "hard",
                        "formula_type": "algebraic",
                        "ground_truth": "h_f = f·(L/D)·(v²/2g)",
                        "domain_specific": True,
                        "constants": {"g": 9.81},
                        "units": {
                            "length": "m",
                            "velocity": "m/s",
                            "diameter": "m",
                            "head_loss": "m",
                        },
                        "extrapolation_test": True,
                    },
                )
            )

            # 2. Reynolds Number
            velocity = np.random.uniform(0.1, 10, num_samples)
            diameter = np.random.uniform(0.01, 1.0, num_samples)
            kinematic_viscosity = np.random.uniform(1e-6, 1e-5, num_samples)  # m²/s
            X = np.column_stack([velocity, diameter, kinematic_viscosity])
            y = velocity * diameter / kinematic_viscosity
            test_cases.append(
                (
                    "Reynolds number for characterizing flow regime",
                    X,
                    y,
                    ["velocity", "characteristic_length", "kinematic_viscosity"],
                    {
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "Re = vL/ν",
                        "domain_specific": True,
                        "units": {
                            "velocity": "m/s",
                            "length": "m",
                            "viscosity": "m²/s",
                        },
                    },
                )
            )

            # 3. Bernoulli's Equation (Simplified)
            pressure1 = np.random.uniform(100000, 200000, num_samples)  # Pa
            velocity1 = np.random.uniform(1, 10, num_samples)  # m/s
            height1 = np.random.uniform(0, 10, num_samples)  # m
            rho = 1000  # kg/m³ (water)
            g = 9.81
            X = np.column_stack([pressure1, velocity1, height1])
            y = pressure1 + 0.5 * rho * velocity1**2 + rho * g * height1
            test_cases.append(
                (
                    "Bernoulli equation: total mechanical energy in fluid flow",
                    X,
                    y,
                    ["pressure", "velocity", "height"],
                    {
                        "difficulty": "medium",
                        "formula_type": "algebraic",
                        "ground_truth": "P + ½ρv² + ρgh = const",
                        "domain_specific": False,
                        "constants": {"rho": 1000, "g": 9.81},
                        "units": {"pressure": "Pa", "velocity": "m/s", "height": "m"},
                    },
                )
            )

            # 4. Hagen-Poiseuille Flow
            pressure_drop = np.random.uniform(1000, 10000, num_samples)  # Pa
            radius = np.random.uniform(0.001, 0.01, num_samples)  # m
            length = np.random.uniform(0.1, 2.0, num_samples)  # m
            viscosity = 0.001  # Pa·s (water)
            X = np.column_stack([pressure_drop, radius, length])
            y = (np.pi * radius**4 * pressure_drop) / (8 * viscosity * length)
            test_cases.append(
                (
                    "Hagen-Poiseuille law: volumetric flow rate in laminar pipe flow",
                    X,
                    y,
                    ["pressure_drop", "pipe_radius", "pipe_length"],
                    {
                        "difficulty": "medium",
                        "formula_type": "power_law",
                        "ground_truth": "Q = πr⁴ΔP/(8μL)",
                        "domain_specific": True,
                        "constants": {"viscosity": 0.001, "pi": np.pi},
                        "units": {
                            "pressure": "Pa",
                            "radius": "m",
                            "length": "m",
                            "flow": "m³/s",
                        },
                    },
                )
            )

        elif domain == "thermodynamics":
            # ==================== THERMODYNAMICS DOMAIN ====================

            # 1. Heat Transfer (Newton's Law of Cooling)
            temp_diff = np.random.uniform(10, 100, num_samples)  # K
            heat_transfer_coeff = np.random.uniform(5, 50, num_samples)  # W/(m²·K)
            area = np.random.uniform(0.1, 10.0, num_samples)  # m²
            X = np.column_stack([heat_transfer_coeff, area, temp_diff])
            y = heat_transfer_coeff * area * temp_diff
            test_cases.append(
                (
                    "Newton's law of cooling: convective heat transfer rate",
                    X,
                    y,
                    [
                        "heat_transfer_coefficient",
                        "surface_area",
                        "temperature_difference",
                    ],
                    {
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "Q = hAΔT",
                        "domain_specific": True,
                        "units": {"h": "W/(m²·K)", "A": "m²", "T": "K", "Q": "W"},
                    },
                )
            )

            # 2. Ideal Gas Law
            pressure = np.random.uniform(100000, 500000, num_samples)  # Pa
            volume = np.random.uniform(0.01, 1.0, num_samples)  # m³
            R = 8.314  # J/(mol·K)
            temperature = np.random.uniform(273, 373, num_samples)  # K
            X = np.column_stack([pressure, volume, temperature])
            y = (pressure * volume) / (R * temperature)
            test_cases.append(
                (
                    "Ideal gas law: number of moles from PVT data",
                    X,
                    y,
                    ["pressure", "volume", "temperature"],
                    {
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "n = PV/(RT)",
                        "domain_specific": False,
                        "constants": {"R": 8.314},
                        "units": {"P": "Pa", "V": "m³", "T": "K", "n": "mol"},
                    },
                )
            )

            # 3. Carnot Efficiency
            T_hot = np.random.uniform(400, 800, num_samples)  # K
            T_cold = np.random.uniform(273, 350, num_samples)  # K
            X = np.column_stack([T_hot, T_cold])
            y = 1 - (T_cold / T_hot)
            test_cases.append(
                (
                    "Carnot efficiency: maximum theoretical efficiency of heat engine",
                    X,
                    y,
                    ["hot_reservoir_temp", "cold_reservoir_temp"],
                    {
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "η = 1 - T_c/T_h",
                        "domain_specific": True,
                        "units": {"T": "K", "efficiency": "fraction"},
                    },
                )
            )

            # 4. Stefan-Boltzmann Law
            temperature = np.random.uniform(300, 1000, num_samples)  # K
            emissivity = np.random.uniform(0.1, 1.0, num_samples)
            area = np.random.uniform(0.1, 10.0, num_samples)  # m²
            sigma = 5.67e-8  # W/(m²·K⁴) Stefan-Boltzmann constant
            X = np.column_stack([emissivity, area, temperature])
            y = emissivity * sigma * area * temperature**4
            test_cases.append(
                (
                    "Stefan-Boltzmann law: radiative heat transfer from black body",
                    X,
                    y,
                    ["emissivity", "surface_area", "temperature"],
                    {
                        "difficulty": "medium",
                        "formula_type": "power_law",
                        "ground_truth": "P = εσAT⁴",
                        "domain_specific": True,
                        "constants": {"sigma": 5.67e-8},
                        "units": {"T": "K", "A": "m²", "P": "W"},
                    },
                )
            )

        elif domain == "mechanics":
            # ==================== MECHANICS DOMAIN ====================

            # 1. Euler Buckling Load
            E = 200e9  # Pa (Young's modulus - steel)
            I = np.random.uniform(1e-8, 1e-6, num_samples)  # m⁴ (moment of inertia)
            L = np.random.uniform(0.5, 5.0, num_samples)  # m (column length)
            X = np.column_stack([I, L])
            y = (np.pi**2 * E * I) / (L**2)
            test_cases.append(
                (
                    "Euler buckling load: critical load for column buckling",
                    X,
                    y,
                    ["moment_of_inertia", "column_length"],
                    {
                        "difficulty": "medium",
                        "formula_type": "power_law",
                        "ground_truth": "P_cr = π²EI/L²",
                        "domain_specific": True,
                        "constants": {"E": 200e9, "pi": np.pi},
                        "units": {"I": "m⁴", "L": "m", "P": "N"},
                    },
                )
            )

            # 2. Hooke's Law (Stress-Strain)
            strain = np.random.uniform(0.001, 0.01, num_samples)
            E = 200e9  # Pa (Young's modulus)
            X = strain.reshape(-1, 1)
            y = E * strain
            test_cases.append(
                (
                    "Hooke's law: linear elastic stress-strain relationship",
                    X,
                    y,
                    ["strain"],
                    {
                        "difficulty": "easy",
                        "formula_type": "linear",
                        "ground_truth": "σ = Eε",
                        "domain_specific": False,
                        "constants": {"E": 200e9},
                        "units": {"strain": "fraction", "stress": "Pa"},
                    },
                )
            )

            # 3. Torsional Stress
            torque = np.random.uniform(100, 10000, num_samples)  # N·m
            radius = np.random.uniform(0.01, 0.1, num_samples)  # m
            J = np.pi * radius**4 / 2  # polar moment of inertia (solid shaft)
            X = np.column_stack([torque, radius])
            y = torque * radius / J
            test_cases.append(
                (
                    "Torsional shear stress in circular shaft",
                    X,
                    y,
                    ["applied_torque", "shaft_radius"],
                    {
                        "difficulty": "medium",
                        "formula_type": "power_law",
                        "ground_truth": "τ = Tr/J",
                        "domain_specific": True,
                        "constants": {"pi": np.pi},
                        "units": {"T": "N·m", "r": "m", "stress": "Pa"},
                    },
                )
            )

            # 4. Bending Stress (Simple Beam)
            moment = np.random.uniform(100, 10000, num_samples)  # N·m
            distance = np.random.uniform(
                0.01, 0.1, num_samples
            )  # m (distance from neutral axis)
            I = np.random.uniform(1e-8, 1e-6, num_samples)  # m⁴ (moment of inertia)
            X = np.column_stack([moment, distance, I])
            y = moment * distance / I
            test_cases.append(
                (
                    "Bending stress in beam under flexure",
                    X,
                    y,
                    [
                        "bending_moment",
                        "distance_from_neutral_axis",
                        "moment_of_inertia",
                    ],
                    {
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "σ = My/I",
                        "domain_specific": True,
                        "units": {"M": "N·m", "y": "m", "I": "m⁴", "stress": "Pa"},
                    },
                )
            )

        elif domain == "chemistry":
            # ==================== CHEMISTRY DOMAIN ====================

            # 1. Arrhenius Equation (Simplified)
            temperature = np.random.uniform(273, 373, num_samples)  # K
            E_a = 50000  # J/mol (activation energy)
            A = 1e10  # pre-exponential factor
            R = 8.314  # J/(mol·K)
            X = temperature.reshape(-1, 1)
            y = A * np.exp(-E_a / (R * temperature))
            test_cases.append(
                (
                    "Arrhenius equation: temperature dependence of reaction rate",
                    X,
                    y,
                    ["temperature"],
                    {
                        "difficulty": "medium",
                        "formula_type": "exponential",
                        "ground_truth": "k = A·exp(-E_a/RT)",
                        "domain_specific": True,
                        "constants": {"A": 1e10, "E_a": 50000, "R": 8.314},
                        "units": {"T": "K", "k": "1/s"},
                    },
                )
            )

            # 2. Henderson-Hasselbalch Equation
            pKa = 4.75  # acetic acid
            conc_acid = np.random.uniform(0.01, 1.0, num_samples)  # M
            conc_base = np.random.uniform(0.01, 1.0, num_samples)  # M
            X = np.column_stack([conc_acid, conc_base])
            y = pKa + np.log10(conc_base / conc_acid)
            test_cases.append(
                (
                    "Henderson-Hasselbalch equation: pH of buffer solution",
                    X,
                    y,
                    ["acid_concentration", "conjugate_base_concentration"],
                    {
                        "difficulty": "medium",
                        "formula_type": "logarithmic",
                        "ground_truth": "pH = pKa + log([A⁻]/[HA])",
                        "domain_specific": True,
                        "constants": {"pKa": 4.75},
                        "units": {"concentration": "M", "pH": "unitless"},
                    },
                )
            )

            # 3. Beer-Lambert Law
            concentration = np.random.uniform(0.001, 0.1, num_samples)  # M
            path_length = np.random.uniform(0.1, 10.0, num_samples)  # cm
            molar_absorptivity = 1000  # L/(mol·cm)
            X = np.column_stack([concentration, path_length])
            y = molar_absorptivity * concentration * path_length
            test_cases.append(
                (
                    "Beer-Lambert law: light absorption in solution",
                    X,
                    y,
                    ["concentration", "path_length"],
                    {
                        "difficulty": "easy",
                        "formula_type": "algebraic",
                        "ground_truth": "A = εcl",
                        "domain_specific": True,
                        "constants": {"epsilon": 1000},
                        "units": {"c": "M", "l": "cm", "A": "absorbance"},
                    },
                )
            )

            # 4. Nernst Equation (Simplified)
            concentration = np.random.uniform(0.001, 1.0, num_samples)  # M
            E_standard = 0.76  # V (standard potential)
            n = 2  # number of electrons
            R = 8.314
            T = 298  # K
            F = 96485  # C/mol (Faraday constant)
            X = concentration.reshape(-1, 1)
            y = E_standard - (R * T / (n * F)) * np.log(1 / concentration)
            test_cases.append(
                (
                    "Nernst equation: electrode potential as function of concentration",
                    X,
                    y,
                    ["ion_concentration"],
                    {
                        "difficulty": "hard",
                        "formula_type": "logarithmic",
                        "ground_truth": "E = E° - (RT/nF)ln(1/[C])",
                        "domain_specific": True,
                        "constants": {
                            "E_0": 0.76,
                            "n": 2,
                            "R": 8.314,
                            "T": 298,
                            "F": 96485,
                        },
                        "units": {"concentration": "M", "potential": "V"},
                    },
                )
            )

        return test_cases

    @staticmethod
    def get_domain_description(domain: str) -> str:
        """Get detailed description of each domain."""
        descriptions = {
            "materials": "Materials Science - grain size effects, porosity, thermal expansion, creep",
            "fluids": "Fluid Dynamics - pipe flow, Reynolds number, Bernoulli, Poiseuille flow",
            "thermodynamics": "Thermodynamics - heat transfer, ideal gas, Carnot efficiency, radiation",
            "mechanics": "Solid Mechanics - buckling, stress-strain, torsion, bending",
            "chemistry": "Chemistry - reaction kinetics, pH, spectroscopy, electrochemistry",
        }
        return descriptions.get(domain, "Unknown domain")

    @staticmethod
    def generate_experiment_report(results: List[Dict]) -> Dict:
        """
        Generate comprehensive experiment report with analysis across domains.

        Args:
            results: List of experiment results

        Returns:
            Dictionary containing detailed analysis
        """
        report = {
            "overall": {},
            "by_domain": {},
            "by_difficulty": {},
            "by_formula_type": {},
            "extrapolation_tests": [],
        }

        # Overall statistics
        total = len(results)
        successful = sum(
            1 for r in results if r.get("evaluation", {}).get("success", False)
        )

        report["overall"]["total_cases"] = total
        report["overall"]["successful"] = successful
        report["overall"]["success_rate"] = successful / total if total > 0 else 0

        # R² statistics for successful cases
        r2_scores = []
        for r in results:
            eval_dict = r.get("evaluation", {})
            if eval_dict.get("success", False) and "r2" in eval_dict:
                r2_scores.append(eval_dict["r2"])

        if r2_scores:
            report["overall"]["mean_r2"] = float(np.mean(r2_scores))
            report["overall"]["median_r2"] = float(np.median(r2_scores))
            report["overall"]["std_r2"] = float(np.std(r2_scores))
            report["overall"]["min_r2"] = float(np.min(r2_scores))
            report["overall"]["max_r2"] = float(np.max(r2_scores))

        # By domain
        domains = set(r["domain"] for r in results)
        for domain in domains:
            domain_results = [r for r in results if r["domain"] == domain]
            domain_successful = sum(
                1
                for r in domain_results
                if r.get("evaluation", {}).get("success", False)
            )
            domain_r2 = [
                r["evaluation"]["r2"]
                for r in domain_results
                if r.get("evaluation", {}).get("success", False)
                and "r2" in r.get("evaluation", {})
            ]

            report["by_domain"][domain] = {
                "total": len(domain_results),
                "successful": domain_successful,
                "success_rate": domain_successful / len(domain_results)
                if len(domain_results) > 0
                else 0,
                "mean_r2": float(np.mean(domain_r2)) if domain_r2 else None,
            }

        # Track extrapolation test cases
        for r in results:
            if (
                r.get("description", "").lower().find("hall-petch") != -1
                or r.get("description", "").lower().find("darcy") != -1
            ):
                report["extrapolation_tests"].append(
                    {
                        "description": r.get("description"),
                        "domain": r.get("domain"),
                        "r2": r.get("evaluation", {}).get("r2"),
                        "rmse": r.get("evaluation", {}).get("rmse"),
                        "success": r.get("evaluation", {}).get("success", False),
                    }
                )

        return report

    @staticmethod
    def save_protocol_documentation(filepath: str = "docs/experiment_protocol.json"):
        """Save complete protocol documentation."""
        protocol_doc = {
            "title": "Pure LLM Formula Discovery - Scientific/Engineering Domains",
            "version": "1.0",
            "date": "2025-01-20",
            "focus": "Materials Science and Fluid Dynamics (Hall-Petch, Darcy-Weisbach)",
            "domains": {},
            "methodology": {
                "approach": "Pure LLM without symbolic regression",
                "model": "Claude Sonnet 4 (claude-sonnet-4-6)",
                "evaluation_metrics": ["R²", "RMSE", "MAE", "MSE"],
                "sample_size": 100,
                "extrapolation": "Test on Hall-Petch and Darcy-Weisbach",
            },
        }

        for domain in ExperimentProtocol.get_all_domains():
            test_cases = ExperimentProtocol.load_test_data(domain, num_samples=10)
            protocol_doc["domains"][domain] = {
                "description": ExperimentProtocol.get_domain_description(domain),
                "num_test_cases": len(test_cases),
                "test_cases": [
                    {
                        "description": desc,
                        "variables": vars,
                        "difficulty": meta["difficulty"],
                        "formula_type": meta["formula_type"],
                        "ground_truth": meta["ground_truth"],
                        "extrapolation_test": meta.get("extrapolation_test", False),
                    }
                    for desc, _, _, vars, meta in test_cases
                ],
            }

        import os

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(protocol_doc, f, indent=2)

        print(f"Protocol documentation saved to: {filepath}")
        return protocol_doc


# Example usage
if __name__ == "__main__":
    protocol = ExperimentProtocol()

    # Print protocol overview
    print("=" * 80)
    print("RESEARCH-ALIGNED EXPERIMENT PROTOCOL".center(80))
    print("Pure LLM Formula Discovery - Scientific/Engineering Domains".center(80))
    print("=" * 80)

    print("\n🎯 KEY TEST CASES:")
    print("  • Hall-Petch (Materials): σ_y = σ_0 + k/√d")
    print("  • Darcy-Weisbach (Fluids): h_f = f·(L/D)·(v²/2g)")

    for domain in protocol.get_all_domains():
        print(f"\n{'=' * 80}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"Description: {protocol.get_domain_description(domain)}")
        test_cases = protocol.load_test_data(domain, num_samples=10)
        print(f"Test cases: {len(test_cases)}")
        print("-" * 80)

        for i, (desc, X, y, vars, meta) in enumerate(test_cases, 1):
            marker = "⭐" if meta.get("extrapolation_test") else "  "
            print(f"{marker} {i}. {desc}")
            print(f"     Variables: {', '.join(vars)}")
            print(
                f"     Difficulty: {meta['difficulty']} | Type: {meta['formula_type']}"
            )
            print(f"     Ground truth: {meta['ground_truth']}")

    # Save documentation
    print("\n" + "=" * 80)
    protocol.save_protocol_documentation()

    print("\n" + "=" * 80)
    total = sum(len(protocol.load_test_data(d)) for d in protocol.get_all_domains())
    print(f"Total test cases: {total}")
    print(f"Extrapolation tests: 2 (Hall-Petch, Darcy-Weisbach)")
    print("=" * 80)
