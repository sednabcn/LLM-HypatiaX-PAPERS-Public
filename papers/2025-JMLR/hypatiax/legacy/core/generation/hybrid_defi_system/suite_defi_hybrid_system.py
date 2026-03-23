#!/usr/bin/env python3
"""
HYPATIAX DEFI HYBRID SYSTEM v4.1 - COMPLETE WITH DETAILED REPORTING
=====================================================================
DeFi-specific implementation with comprehensive results export

NEW IN v4.1:
✅ Detailed results table at end of run (Test Case | R² | Val | Observations)
✅ Complete JSON export with ALL test outputs (complete_results.json)
✅ Enhanced observations and categorization
✅ Statistical analysis (mean, median, std, min, max)
✅ 20 DeFi test cases across 6 domains
✅ Resume capability with checkpoints
✅ Custom --iterations support
✅ Fast/Standard/Thorough modes
✅ Session management
✅ 6 extrapolation tests

Usage:
    # Run all DeFi tests
    python suite_defi_hybrid_system.py --batch

    # Resume interrupted run
    python suite_defi_hybrid_system.py --batch --resume

    # Custom iterations
    python suite_defi_hybrid_system.py --batch --iterations 50

    # Run specific domain
    python suite_defi_hybrid_system.py --domain staking --batch

    # Single test
    python suite_defi_hybrid_system.py --test compound_staking

Author: HypatiaX Team
Version: 4.1 Enhanced Edition
Date: 2026-01-07
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from hypatiax.tools.symbolic.hybrid_system_v40 import HybridDiscoverySystem

    HYBRID_VERSION = "v4.0 (Auto-Config)"
except ImportError:
     print("❌ Error: hybrid_system_v40.py not found in current directory")
    sys.exit(1)
    
from hypatiax.tools.validation.ensemble_validator import EnsembleValidator

# Import DeFi protocol
try:
    from hypatiax.protocols.experiment_protocol_defi_20 import DeFiExperimentProtocolExtended
except ImportError:
    print("❌ Error: experiment_protocol_defi_20.py not found in current directory")
    sys.exit(1)

import os

os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"
os.environ["PYTHON_JULIACALL_QUIET"] = "yes"
os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("hypatiax/data/results/defi")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SESSION_FILE = RESULTS_DIR / "current_session.json"

# FAST MODE: Development/testing (~1-2 min per test)
FAST_CONFIG = {
    "niterations": 20,
    "populations": 8,
    "enable_auto_configuration": True,
    "auto_config_correlation_threshold": 0.15,
}

# STANDARD MODE: Production (~5-8 min per test)
STANDARD_CONFIG = {
    "niterations": 50,
    "populations": 12,
    "enable_auto_configuration": True,
    "auto_config_correlation_threshold": 0.2,
}

# THOROUGH MODE: Final validation (~15-20 min per test)
THOROUGH_CONFIG = {
    "niterations": 100,
    "populations": 15,
    "enable_auto_configuration": True,
    "auto_config_correlation_threshold": 0.2,
}

SYMBOLIC_CONFIG = FAST_CONFIG

# ============================================================================
# SESSION MANAGEMENT (ENHANCED)
# ============================================================================


class SessionManager:
    """Manages test session with resume capability and complete results export."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = RESULTS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.session_dir / "checkpoint.json"
        self.completed_tests = set()
        self.failed_tests = set()

        self._load_checkpoint()

    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    self.completed_tests = set(data.get("completed", []))
                    self.failed_tests = set(data.get("failed", []))
                    print(f"\n📂 Loaded checkpoint:")
                    print(f"   Completed: {len(self.completed_tests)} tests")
                    print(f"   Failed: {len(self.failed_tests)} tests")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")

    def _save_checkpoint(self):
        data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "completed": list(self.completed_tests),
            "failed": list(self.failed_tests),
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(data, f, indent=2)

    def is_completed(self, test_name: str) -> bool:
        return test_name in self.completed_tests

    def save_test_result(self, test_name: str, result: Dict, passed: bool):
        test_file = self.session_dir / f"{test_name}.json"

        result["_metadata"] = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "passed": passed,
            "test_name": test_name,
        }

        # Make JSON-serializable
        clean_result = self._clean_for_json(result)

        with open(test_file, "w") as f:
            json.dump(clean_result, f, indent=2, default=str)

        if passed:
            self.completed_tests.add(test_name)
        else:
            self.failed_tests.add(test_name)

        self._save_checkpoint()
        print(f"   💾 Saved: {test_file.name}")

    def _clean_for_json(self, obj):
        """Recursively clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj

    def load_all_results(self) -> Dict[str, Dict]:
        results = {}
        for test_file in self.session_dir.glob("*.json"):
            if test_file.name in [
                "checkpoint.json",
                "summary.json",
                "complete_results.json",
            ]:
                continue

            try:
                with open(test_file, "r") as f:
                    data = json.load(f)
                    test_name = test_file.stem
                    results[test_name] = data
            except Exception as e:
                print(f"⚠️  Could not load {test_file.name}: {e}")

        return results

    def save_summary(self, summary: Dict):
        """Save summary with complete JSON export of all results."""
        # Save standard summary
        summary_file = self.session_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n📊 Summary saved: {summary_file}")

        # NEW: Save complete results export
        complete_export = self._generate_complete_export(summary)
        complete_file = self.session_dir / "complete_results.json"
        with open(complete_file, "w") as f:
            json.dump(complete_export, f, indent=2, default=str)
        print(f"📦 Complete results: {complete_file}")
        print(f"   Size: {complete_file.stat().st_size / 1024:.1f} KB")

    def _generate_complete_export(self, summary: Dict) -> Dict:
        """Generate complete JSON export with all test outputs."""
        all_results = self.load_all_results()

        export = {
            "session_metadata": {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "configuration": summary.get("configuration", {}),
                "total_tests": summary.get("total_tests", 0),
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "pass_rate": (
                    summary.get("passed", 0) / max(summary.get("total_tests", 1), 1)
                )
                * 100,
                "total_time_minutes": summary.get("total_time", 0) / 60,
            },
            "summary_statistics": {
                "by_domain": dict(summary.get("by_domain", {})),
                "by_difficulty": dict(summary.get("by_difficulty", {})),
                "extrapolation_tests": summary.get("extrapolation_results", []),
                "detailed_results_table": summary.get("detailed_results", []),
            },
            "individual_test_results": {},
        }

        # Add each test's complete output
        for test_name, result in all_results.items():
            # Organize into structured format
            test_export = {
                "metadata": {
                    "test_name": test_name,
                    "domain": result.get("domain", "unknown"),
                    "difficulty": result.get("difficulty", "unknown"),
                    "ground_truth": result.get("ground_truth", "N/A"),
                    "extrapolation_test": result.get("extrapolation_test", False),
                    "timestamp": result.get("timestamp", "unknown"),
                    "execution_time_seconds": result.get("execution_time", 0.0),
                },
                "discovery": result.get("discovery", {}),
                "validation": result.get("validation", {}),
                "interpretation": result.get("interpretation", {}),
                "variables": {
                    "names": result.get("variable_names", []),
                    "units": result.get("variable_units", {}),
                    "descriptions": result.get("variable_descriptions", {}),
                },
                "error": result.get("error"),
                "full_result": result,  # Complete original result
            }

            export["individual_test_results"][test_name] = test_export

        return export

    def get_pending_tests(self, all_tests: List[str]) -> List[str]:
        return [t for t in all_tests if t not in self.completed_tests]

    def print_status(self, all_tests: List[str]):
        total = len(all_tests)
        completed = len(self.completed_tests)
        pending = total - completed

        print(f"\n📊 Session Status:")
        print(f"   Session ID: {self.session_id}")
        print(f"   Total tests: {total}")
        print(f"   ✅ Completed: {completed}")
        print(f"   ⏳ Pending: {pending}")
        if self.failed_tests:
            print(f"   ❌ Failed: {len(self.failed_tests)}")
        print(f"   Results dir: {self.session_dir}")


# ============================================================================
# PROTOCOL TO TEST CASES CONVERTER
# ============================================================================


def convert_defi_protocol_to_test_cases(
    protocol: DeFiExperimentProtocolExtended, domains: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """Convert DeFi protocol to internal test case format."""

    test_cases = {}
    all_domains = protocol.get_all_domains()
    domains_to_load = domains if domains else all_domains

    print(f"\n📥 Converting DeFi Protocol to test cases...")
    print(f"   Domains: {', '.join(domains_to_load)}")

    for domain in domains_to_load:
        if domain not in all_domains:
            print(f"⚠️  Domain '{domain}' not found, skipping...")
            continue

        protocol_tests = protocol.load_test_data(domain, num_samples=100)

        for desc, X_sample, y_sample, var_names, metadata in protocol_tests:
            eq_name = metadata.get("equation_name", "unknown")
            test_name = f"{domain}_{eq_name}"

            # Create proper data generator closure
            def make_generator(proto, dom, eq):
                def generator(n):
                    tests = proto.load_test_data(dom, num_samples=n)
                    for d, X, y, v, m in tests:
                        if m.get("equation_name") == eq:
                            return X, lambda arr: y
                    raise ValueError(f"Test {eq} not found in protocol")

                return generator

            # Extract units from metadata if available
            units = metadata.get("units", {})
            if not units:
                # Create default units
                units = {var: "dimensionless" for var in var_names}

            test_cases[test_name] = {
                "domain": domain,
                "equation_name": eq_name,
                "name": metadata.get("equation_name", desc).replace("_", " ").title(),
                "description": desc,
                "ground_truth": metadata.get("ground_truth", ""),
                "variables": var_names,
                "variable_descriptions": {var: f"{var} in {desc}" for var in var_names},
                "variable_units": units,
                "metadata": metadata,
                "protocol": "DeFi_Protocol_v3.0",
                "generate_data": make_generator(protocol, domain, eq_name),
                "use_enhanced_config": metadata.get("use_enhanced_config", False),
                "extrapolation_test": metadata.get("extrapolation_test", False),
                "difficulty": metadata.get("difficulty", "medium"),
            }

    print(f"✅ Converted {len(test_cases)} test cases")

    # Print extrapolation tests
    extrap_tests = [
        name for name, tc in test_cases.items() if tc.get("extrapolation_test")
    ]
    if extrap_tests:
        print(f"\n🚀 Extrapolation tests ({len(extrap_tests)}):")
        for name in extrap_tests:
            print(f"   - {name}")

    return test_cases


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def print_header(title: str, width: int = 80):
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def get_config_name() -> str:
    if SYMBOLIC_CONFIG == FAST_CONFIG:
        return "FAST"
    elif SYMBOLIC_CONFIG == STANDARD_CONFIG:
        return "STANDARD"
    elif SYMBOLIC_CONFIG == THOROUGH_CONFIG:
        return "THOROUGH"
    return f"CUSTOM (iter={SYMBOLIC_CONFIG['niterations']})"


def list_test_cases_by_domain(test_cases: Dict):
    print_header(f"DEFI TEST CASES BY DOMAIN ({len(test_cases)} TOTAL)")

    print(f"\n🔧 Hybrid System Version: {HYBRID_VERSION}")
    print(f"⚙️  Configuration Mode: {get_config_name()}")
    print(f"   Iterations: {SYMBOLIC_CONFIG['niterations']}")
    print(f"   Populations: {SYMBOLIC_CONFIG['populations']}")

    domains = set(case["domain"] for case in test_cases.values())

    for domain in sorted(domains):
        cases = {
            name: case for name, case in test_cases.items() if case["domain"] == domain
        }
        print(f"\n{domain.upper()} ({len(cases)} tests):")

        for name, case in cases.items():
            difficulty = case.get("difficulty", "?")
            extrap = " 🚀" if case.get("extrapolation_test") else ""
            enhanced = " ⚡" if case.get("use_enhanced_config") else ""
            print(f"  [{difficulty:6s}] {name:35s}{extrap}{enhanced}")

    print(f"\n{'=' * 80}")
    print(f"Total: {len(test_cases)} test cases across {len(domains)} domains")

    # Count by difficulty
    easy = sum(1 for tc in test_cases.values() if tc.get("difficulty") == "easy")
    medium = sum(1 for tc in test_cases.values() if tc.get("difficulty") == "medium")
    hard = sum(1 for tc in test_cases.values() if tc.get("difficulty") == "hard")
    extrap = sum(1 for tc in test_cases.values() if tc.get("extrapolation_test"))

    print(f"\nDifficulty: Easy: {easy} | Medium: {medium} | Hard: {hard}")
    print(f"Extrapolation: {extrap} tests")
    print(f"{'=' * 80}")


# ============================================================================
# TEST EXECUTION
# ============================================================================


def run_single_test(
    test_name: str,
    test_cases: Dict,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    verbose: bool = True,
    session: Optional[SessionManager] = None,
) -> Dict:
    """Run a single DeFi test case."""

    if test_name not in test_cases:
        raise ValueError(f"Unknown test: {test_name}")

    test_config = test_cases[test_name]

    if verbose:
        print_header(f"Running: {test_config['name']}", 80)
        print(f"Domain: {test_config['domain']}")
        print(f"Description: {test_config['description']}")
        print(f"Variables: {', '.join(test_config['variables'])}")
        print(f"Difficulty: {test_config.get('difficulty', 'unknown')}")
        print(f"Ground Truth: {test_config.get('ground_truth', 'N/A')}")
        if test_config.get("extrapolation_test"):
            print(f"🚀 EXTRAPOLATION TEST")
        print(f"Mode: {get_config_name()} (iter={SYMBOLIC_CONFIG['niterations']})")

    start_time = time.time()

    try:
        # Generate data
        if seed is not None:
            np.random.seed(seed)

        X, y_func = test_config["generate_data"](n_samples)
        y = y_func(X)

        # Import DiscoveryConfig
        from hypatiax.tools.symbolic.symbolic_engine import DiscoveryConfig

        # Create DiscoveryConfig
        discovery_config = DiscoveryConfig(
            niterations=SYMBOLIC_CONFIG["niterations"],
            populations=SYMBOLIC_CONFIG["populations"],
            enable_auto_configuration=SYMBOLIC_CONFIG["enable_auto_configuration"],
            auto_config_correlation_threshold=SYMBOLIC_CONFIG[
                "auto_config_correlation_threshold"
            ],
        )

        # Initialize hybrid system
        hybrid = HybridDiscoverySystem(
            domain=test_config["domain"],
            discovery_config=discovery_config,
            enable_auto_config=SYMBOLIC_CONFIG["enable_auto_configuration"],
            max_retries=5,
            enable_physics_fallback=False,
            anthropic_api_key=None,
            google_api_key=None,
        )

        if verbose:
            print(f"\n🔬 Starting discovery...")

        # Run discovery
        result = hybrid.discover_validate_interpret(
            X=X,
            y=y,
            variable_names=test_config["variables"],
            variable_descriptions=test_config.get("variable_descriptions", {}),
            variable_units=test_config.get("variable_units", {}),
            description=test_config.get("name", test_name),
            equation_name=test_config.get("equation_name"),
            validate_first=True,
        )

        # Add metadata
        result["n_samples"] = n_samples
        result["execution_time"] = time.time() - start_time
        result["test_name"] = test_name
        result["timestamp"] = datetime.now().isoformat()
        result["ground_truth"] = test_config.get("ground_truth", "")
        result["domain"] = test_config["domain"]
        result["difficulty"] = test_config.get("difficulty", "unknown")
        result["extrapolation_test"] = test_config.get("extrapolation_test", False)
        result["metadata"] = test_config.get("metadata", {})
        result["variable_units"] = test_config.get("variable_units", {})

        # Determine pass/fail
        discovery = result.get("discovery", {})
        validation = result.get("validation", {})

        discovery_r2 = discovery.get("r2_score", 0.0)
        val_score = validation.get("total_score", validation.get("overall_score", 0.0))
        val_passed = validation.get("valid", False)
        expr = discovery.get("expression")

        # Enhanced pass criteria for DeFi
        passed = (
            (discovery_r2 > 0.99 and val_score > 30.0)
            or (discovery_r2 > 0.95 and val_score > 80.0)
            or val_passed
        )

        # Save immediately
        if session:
            session.save_test_result(test_name, result, passed)

        if verbose:
            print(f"\n📊 Quick Results:")
            print(f"   Expression: {expr}")
            print(f"   R²: {discovery_r2:.4f}")
            print(f"   Validation: {val_score:.1f}/100")
            print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
            print(f"   Time: {result['execution_time']:.1f}s")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        error_result = {
            "error": str(e),
            "test_name": test_name,
            "execution_time": elapsed,
            "n_samples": n_samples,
            "timestamp": datetime.now().isoformat(),
            "domain": test_config["domain"],
            "ground_truth": test_config.get("ground_truth", ""),
            "difficulty": test_config.get("difficulty", "unknown"),
            "extrapolation_test": test_config.get("extrapolation_test", False),
        }

        if session:
            session.save_test_result(test_name, error_result, False)

        if verbose:
            print(f"\n❌ Error: {str(e)}")

        return error_result


def run_all_tests_with_resume(
    test_cases: Dict,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    verbose: bool = True,
    resume: bool = False,
    skip_tests: List[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Dict]:
    """Run all DeFi tests with resume capability."""

    # Initialize session
    if resume and SESSION_FILE.exists():
        with open(SESSION_FILE, "r") as f:
            session_data = json.load(f)
            session_id = session_data.get("session_id")

    session = SessionManager(session_id)

    # Save current session
    with open(SESSION_FILE, "w") as f:
        json.dump({"session_id": session.session_id}, f)

    print_header("DEFI HYBRID SYSTEM WITH RESUME", 80)

    # Get test list
    all_test_names = list(test_cases.keys())

    # Apply skip list
    if skip_tests:
        all_test_names = [t for t in all_test_names if t not in skip_tests]
        print(f"\n⭐️  Skipping: {', '.join(skip_tests)}")

    # Show session status
    session.print_status(all_test_names)

    # Get pending tests
    if resume:
        pending_tests = session.get_pending_tests(all_test_names)
        if not pending_tests:
            print(f"\n✅ All tests already completed!")
            return session.load_all_results()
        print(f"\n🔄 Resuming from checkpoint...")
        print(f"   Remaining: {', '.join(pending_tests)}")
    else:
        pending_tests = all_test_names

    print(f"\n🔧 Configuration:")
    print(f"   Mode: {get_config_name()}")
    print(f"   Samples per test: {n_samples}")
    print(f"   Tests to run: {len(pending_tests)}/{len(all_test_names)}")
    print(f"   Iterations: {SYMBOLIC_CONFIG['niterations']}")
    print(f"   Populations: {SYMBOLIC_CONFIG['populations']}")

    # Estimate time
    if get_config_name() == "FAST":
        est_time = len(pending_tests) * 1.5
    elif get_config_name() == "STANDARD":
        est_time = len(pending_tests) * 6.5
    elif "CUSTOM" in get_config_name():
        est_time = len(pending_tests) * (SYMBOLIC_CONFIG["niterations"] * 0.06)
    else:
        est_time = len(pending_tests) * 17.5
    print(f"   Estimated time: ~{est_time:.0f} min")

    start_time = time.time()

    for i, test_name in enumerate(pending_tests, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}/{len(pending_tests)}: {test_name}")
        print(f"{'=' * 80}")

        try:
            result = run_single_test(
                test_name=test_name,
                test_cases=test_cases,
                n_samples=n_samples,
                seed=seed,
                verbose=verbose,
                session=session,
            )

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            print(f"💾 Progress saved. Resume with: --resume")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            continue

    total_time = time.time() - start_time

    # Load all results
    results = session.load_all_results()

    # Generate summary
    summary = generate_summary(results, total_time, test_cases)
    session.save_summary(summary)

    print_summary(summary)

    return results


# ============================================================================
# SUMMARY GENERATION (ENHANCED)
# ============================================================================


def generate_summary(
    results: Dict[str, Dict], total_time: float, test_cases: Dict
) -> Dict:
    """Generate DeFi test summary statistics with detailed results."""

    summary = {
        "total_tests": len(results),
        "total_time": total_time,
        "passed": 0,
        "failed": 0,
        "by_domain": defaultdict(
            lambda: {"passed": 0, "failed": 0, "extrapolation": 0}
        ),
        "by_difficulty": defaultdict(lambda: {"passed": 0, "failed": 0}),
        "extrapolation_results": [],
        "detailed_results": [],  # NEW: Store detailed results for table
        "configuration": {
            "mode": get_config_name(),
            "iterations": SYMBOLIC_CONFIG["niterations"],
            "populations": SYMBOLIC_CONFIG["populations"],
        },
    }

    for test_name, result in results.items():
        metadata = result.get("_metadata", {})
        passed = metadata.get("passed", False)

        if passed:
            summary["passed"] += 1
        else:
            summary["failed"] += 1

        # By domain
        domain = result.get("domain", "unknown")
        if passed:
            summary["by_domain"][domain]["passed"] += 1
        else:
            summary["by_domain"][domain]["failed"] += 1

        # Track extrapolation tests
        if result.get("extrapolation_test"):
            summary["by_domain"][domain]["extrapolation"] += 1
            summary["extrapolation_results"].append(
                {
                    "test_name": test_name,
                    "domain": domain,
                    "passed": passed,
                    "r2": result.get("discovery", {}).get("r2_score", 0.0),
                }
            )

        # By difficulty
        difficulty = result.get("difficulty", "unknown")
        if passed:
            summary["by_difficulty"][difficulty]["passed"] += 1
        else:
            summary["by_difficulty"][difficulty]["failed"] += 1

        # NEW: Add to detailed results table
        discovery = result.get("discovery", {})
        validation = result.get("validation", {})

        detailed_entry = {
            "test_name": test_name,
            "domain": domain,
            "difficulty": difficulty,
            "r2": discovery.get("r2_score", 0.0),
            "validation_score": validation.get(
                "total_score", validation.get("overall_score", 0.0)
            ),
            "time": result.get("execution_time", 0.0),
            "passed": passed,
            "extrapolation": result.get("extrapolation_test", False),
            "expression": discovery.get("expression", "N/A"),
            "ground_truth": result.get("ground_truth", "N/A"),
            "error": result.get("error"),
        }

        summary["detailed_results"].append(detailed_entry)

    # Sort detailed results by domain, then by test name
    summary["detailed_results"].sort(key=lambda x: (x["domain"], x["test_name"]))

    return summary


def print_summary(summary: Dict):
    """Print formatted DeFi test summary with detailed results table."""

    print_header("DEFI TEST SUITE SUMMARY", 80)

    total = summary["total_tests"]
    passed = summary["passed"]
    failed = summary["failed"]
    pass_rate = (passed / total * 100) if total > 0 else 0

    print(f"\n✅ Passed: {passed}/{total} ({pass_rate:.1f}%)")
    print(f"❌ Failed: {failed}/{total}")
    print(f"⏱️  Total time: {summary['total_time'] / 60:.1f} min")
    print(f"📊 Configuration: {summary['configuration']['mode']}")
    print(f"   Iterations: {summary['configuration']['iterations']}")
    print(f"   Populations: {summary['configuration']['populations']}")

    if summary.get("by_domain"):
        print(f"\n📦 By Domain:")
        for domain, stats in summary["by_domain"].items():
            total_domain = stats["passed"] + stats["failed"]
            rate = (stats["passed"] / total_domain * 100) if total_domain > 0 else 0
            extrap = (
                f" ({stats['extrapolation']} extrap)"
                if stats["extrapolation"] > 0
                else ""
            )
            print(
                f"  {domain:20s}: {stats['passed']:2d}/{total_domain:2d} ({rate:5.1f}%){extrap}"
            )

    if summary.get("by_difficulty"):
        print(f"\n🎯 By Difficulty:")
        for diff, stats in summary["by_difficulty"].items():
            total_diff = stats["passed"] + stats["failed"]
            rate = (stats["passed"] / total_diff * 100) if total_diff > 0 else 0
            print(f"  {diff:10s}: {stats['passed']:2d}/{total_diff:2d} ({rate:5.1f}%)")

    if summary.get("extrapolation_results"):
        print(f"\n🚀 Extrapolation Tests ({len(summary['extrapolation_results'])}):")
        for result in summary["extrapolation_results"]:
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {result['test_name']:35s} R²={result['r2']:.4f}")

    # ========================================================================
    # NEW: DETAILED RESULTS TABLE
    # ========================================================================
    if summary.get("detailed_results"):
        print(f"\n{'=' * 120}")
        print(f"DETAILED TEST RESULTS".center(120))
        print(f"{'=' * 120}")
        print(
            f"{'Test Case':<40} | {'R²':>6} | {'Val':>5} | {'Time':>6} | {'Status':>6} | {'Observations':<40}"
        )
        print(f"{'-' * 120}")

        for result in summary["detailed_results"]:
            test_name = result["test_name"][:38]
            r2 = result.get("r2", 0.0)
            val_score = result.get("validation_score", 0.0)
            time_sec = result.get("time", 0.0)
            status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"

            # Generate observations
            observations = []
            if result.get("extrapolation"):
                observations.append("Extrap")
            if r2 >= 0.99:
                observations.append("Excellent fit")
            elif r2 >= 0.95:
                observations.append("Good fit")
            elif r2 >= 0.90:
                observations.append("Acceptable")
            else:
                observations.append("Poor fit")

            if val_score >= 85:
                observations.append("Strong val")
            elif val_score >= 70:
                observations.append("OK val")
            elif val_score >= 50:
                observations.append("Weak val")
            else:
                observations.append("Failed val")

            if result.get("error"):
                observations.append("ERROR")

            obs_str = ", ".join(observations)[:38]

            print(
                f"{test_name:<40} | {r2:>6.4f} | {val_score:>5.1f} | {time_sec:>5.1f}s | {status:>6} | {obs_str:<40}"
            )

        print(f"{'=' * 120}")

        # Summary statistics
        r2_values = [
            r["r2"]
            for r in summary["detailed_results"]
            if "r2" in r and not r.get("error")
        ]
        val_values = [
            r["validation_score"]
            for r in summary["detailed_results"]
            if "validation_score" in r and not r.get("error")
        ]

        if r2_values:
            print(f"\n📈 R² Statistics:")
            print(f"   Mean: {np.mean(r2_values):.4f}")
            print(f"   Median: {np.median(r2_values):.4f}")
            print(f"   Std Dev: {np.std(r2_values):.4f}")
            print(f"   Min: {np.min(r2_values):.4f}")
            print(f"   Max: {np.max(r2_values):.4f}")

        if val_values:
            print(f"\n📊 Validation Statistics:")
            print(f"   Mean: {np.mean(val_values):.1f}/100")
            print(f"   Median: {np.median(val_values):.1f}/100")
            print(f"   Std Dev: {np.std(val_values):.1f}")
            print(f"   Min: {np.min(val_values):.1f}/100")
            print(f"   Max: {np.max(val_values):.1f}/100")

    print(f"\n{'=' * 80}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="HypatiaX DeFi Hybrid System Test Suite v4.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all DeFi tests
  python suite_defi_hybrid_system.py --batch
  
  # Resume interrupted run
  python suite_defi_hybrid_system.py --batch --resume
  
  # Custom iterations
  python suite_defi_hybrid_system.py --batch --iterations 50
  
  # Run specific domain
  python suite_defi_hybrid_system.py --domain staking --batch
  
  # Run single test
  python suite_defi_hybrid_system.py --test compound_staking
  
  # List all tests
  python suite_defi_hybrid_system.py --list
        """,
    )

    # Test selection
    parser.add_argument("--test", type=str, help="Run specific test by name")
    parser.add_argument(
        "--domain",
        type=str,
        choices=[
            "amm",
            "risk_var",
            "liquidity",
            "expected_shortfall",
            "liquidation",
            "staking",
        ],
        help="Run all tests in a specific domain",
    )
    parser.add_argument("--batch", action="store_true", help="Run all tests")
    parser.add_argument("--skip", type=str, nargs="+", help="Skip specific tests")

    # Configuration
    parser.add_argument(
        "--mode",
        type=str,
        choices=["FAST", "STANDARD", "THOROUGH"],
        default="FAST",
        help="Configuration mode (default: FAST)",
    )
    parser.add_argument(
        "--iterations", type=int, help="Custom number of iterations (overrides mode)"
    )
    parser.add_argument("--populations", type=int, help="Custom population size")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples per test (default: 1000)",
    )
    parser.add_argument("--seed", type=int, help="Random seed")

    # Resume capability
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument("--session-id", type=str, help="Specific session ID to resume")
    parser.add_argument(
        "--force", action="store_true", help="Force rerun even if test completed"
    )

    # Information
    parser.add_argument("--list", action="store_true", help="List all available tests")
    parser.add_argument(
        "--list-sessions", action="store_true", help="List previous test sessions"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Handle --list-sessions
    if args.list_sessions:
        print_header("PREVIOUS DEFI TEST SESSIONS", 80)
        sessions = sorted(RESULTS_DIR.glob("20*"), reverse=True)
        if not sessions:
            print("\nNo previous sessions found.")
        else:
            for session_dir in sessions[:10]:
                checkpoint = session_dir / "checkpoint.json"
                if checkpoint.exists():
                    with open(checkpoint, "r") as f:
                        data = json.load(f)
                        completed = len(data.get("completed", []))
                        failed = len(data.get("failed", []))
                        timestamp = data.get("timestamp", "unknown")
                    print(f"\n📁 {session_dir.name}")
                    print(f"   ✅ Completed: {completed}")
                    print(f"   ❌ Failed: {failed}")
                    print(f"   🕐 Time: {timestamp}")
                    print(f"   Resume: --session-id {session_dir.name}")
        return

    # Apply configuration mode
    global SYMBOLIC_CONFIG
    if args.mode == "FAST":
        SYMBOLIC_CONFIG = FAST_CONFIG.copy()
    elif args.mode == "STANDARD":
        SYMBOLIC_CONFIG = STANDARD_CONFIG.copy()
    elif args.mode == "THOROUGH":
        SYMBOLIC_CONFIG = THOROUGH_CONFIG.copy()

    # Override with custom iterations
    if args.iterations is not None:
        SYMBOLIC_CONFIG["niterations"] = args.iterations
        print(f"\n⚙️  Custom iterations: {args.iterations}")

    # Override with custom populations
    if args.populations is not None:
        SYMBOLIC_CONFIG["populations"] = args.populations
        print(f"⚙️  Custom populations: {args.populations}")

    # Load DeFi protocol
    print(f"\n📄 Loading DeFi Protocol v3.0...")
    protocol = DeFiExperimentProtocolExtended()

    # Show protocol statistics
    stats = protocol.get_protocol_statistics()
    print(f"\n{'=' * 80}")
    print(f"SUMMARY".center(80))
    print(f"{'=' * 80}")
    print(f"Total tests: {stats['total_tests']}")
    print(f"Extrapolation tests: {stats['extrapolation_tests']}")
    print(f"Domains: {len(stats['domains'])}")
    print(f"Difficulty breakdown:")
    print(f"  Easy: {stats['difficulty']['easy']}")
    print(f"  Medium: {stats['difficulty']['medium']}")
    print(f"  Hard: {stats['difficulty']['hard']}")
    print(f"{'=' * 80}")

    # Convert to test cases
    domains = [args.domain] if args.domain else None
    test_cases = convert_defi_protocol_to_test_cases(protocol, domains)

    if not test_cases:
        print("\n❌ No test cases loaded")
        return

    # Handle --list
    if args.list:
        list_test_cases_by_domain(test_cases)
        return

    # Run tests
    verbose = not args.quiet

    if args.test:
        # Single test
        if args.test not in test_cases:
            print(f"\n❌ Test '{args.test}' not found")
            print("\n📋 Available tests:")
            for name in sorted(test_cases.keys()):
                print(f"  - {name}")
            return

        session = SessionManager(args.session_id)

        if not args.force and session.is_completed(args.test):
            print(f"\n✅ Test '{args.test}' already completed")
            print(f"   Use --force to rerun")
            return

        result = run_single_test(
            test_name=args.test,
            test_cases=test_cases,
            n_samples=args.samples,
            seed=args.seed,
            verbose=verbose,
            session=session,
        )

    elif args.batch:
        # Batch mode
        results = run_all_tests_with_resume(
            test_cases=test_cases,
            n_samples=args.samples,
            seed=args.seed,
            verbose=verbose,
            resume=args.resume and not args.force,
            skip_tests=args.skip,
            session_id=args.session_id,
        )

    else:
        parser.print_help()
        print("\n❌ Error: Specify --test, --batch, --list, or --list-sessions")


if __name__ == "__main__":
    main()


"""
USAGE
# Run immediately
# python suite_defi_hybrid_system.py --batch --mode FAST

# Run single test
#python suite_defi_hybrid_system.py --test amm_impermanent_loss

# Resume capability
#python suite_defi_hybrid_system.py --batch --resume

# List tests
#python suite_defi_hybrid_system.py --list

"""
