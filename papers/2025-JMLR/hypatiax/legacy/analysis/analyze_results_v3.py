#!/usr/bin/env python3
"""
HYPATIAX RESULTS ANALYZER
=========================
Comprehensive analysis tool for HypatiaX test results JSON files

Usage:
    python analyze_results.py <json_file>
    python analyze_results.py --compare run1.json run2.json run3.json
    python analyze_results.py --file-list json_files.txt
    python analyze_results.py --plot-failures <directory>
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import argparse
import re
from datetime import datetime

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class ResultsAnalyzer:
    """Comprehensive analyzer for HypatiaX test results"""

    def __init__(self, json_path: str):
        """Initialize analyzer with a single JSON file path"""
        self.json_path = Path(json_path)

        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        try:
            with open(self.json_path, "r") as f:
                self.data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.json_path}: {e}")

        self.results = self.data.get("results", {})
        self.summary = self.data.get("summary", {})

        # Analysis storage
        self.test_metrics = {}
        self.failures = []
        self.failure_categories = defaultdict(list)
        self.r2_scores = {}
        self.validation_scores = {}

    def analyze_all(self):
        """Run complete analysis"""
        print(f"\n{'=' * 80}")
        print(f"ANALYZING: {self.json_path.name}")
        print(f"{'=' * 80}\n")

        self._extract_metrics()
        self._categorize_failures()
        self._print_summary()
        self._print_r2_analysis()
        self._print_failure_analysis()
        self._print_consistency_report()
        self._print_recommendations()

    def _extract_metrics(self):
        """Extract all relevant metrics from results"""
        for test_name, result in self.results.items():
            metrics = {
                "test_name": test_name,
                "domain": result.get("domain", "unknown"),
                "r2_score": None,
                "validation_score": None,
                "is_valid": False,
                "discovery_engine": result.get("metadata", {}).get(
                    "discovery_engine", "unknown"
                ),
                "errors": [],
                "warnings": [],
                "discovered_expr": None,
                "fit_time": None,
            }

            # Extract R² score
            discovery = result.get("discovery", {})
            metrics["r2_score"] = discovery.get("r2_score", 0.0)
            metrics["discovered_expr"] = discovery.get("expression", "N/A")
            metrics["fit_time"] = discovery.get("fit_time", 0.0)

            # Extract validation metrics
            validation = result.get("validation", {})
            metrics["is_valid"] = validation.get("valid", False)
            metrics["validation_score"] = validation.get("total_score", 0.0)
            metrics["errors"] = validation.get("errors", [])
            metrics["warnings"] = validation.get("warnings", [])

            # Layer scores
            layer_results = validation.get("layer_results", {})
            metrics["symbolic_score"] = layer_results.get("symbolic", {}).get(
                "score", 0
            )
            metrics["dimensional_score"] = layer_results.get("dimensional", {}).get(
                "score", 0
            )
            metrics["domain_score"] = layer_results.get("domain", {}).get("score", 0)
            metrics["numerical_score"] = layer_results.get("numerical", {}).get(
                "score", 0
            )

            self.test_metrics[test_name] = metrics
            self.r2_scores[test_name] = metrics["r2_score"]
            self.validation_scores[test_name] = metrics["validation_score"]

            # Track failures
            if not metrics["is_valid"] or metrics["validation_score"] < 85.0:
                self.failures.append(test_name)

    def _categorize_failures(self):
        """Categorize failures by root cause"""
        for test_name in self.failures:
            metrics = self.test_metrics[test_name]
            errors = metrics["errors"]

            # Category 1: Discovery failure
            if "DISCOVERY_FAILED" in str(metrics["discovered_expr"]):
                self.failure_categories["discovery_failure"].append(
                    {
                        "test": test_name,
                        "domain": metrics["domain"],
                        "r2": metrics["r2_score"],
                        "reason": "Discovery engine failed to find expression",
                        "errors": errors,
                    }
                )

            # Category 2: Dimensional inconsistency
            elif any("Incompatible units" in str(e) for e in errors):
                dim_errors = [e for e in errors if "Incompatible units" in str(e)]
                self.failure_categories["dimensional_inconsistency"].append(
                    {
                        "test": test_name,
                        "domain": metrics["domain"],
                        "r2": metrics["r2_score"],
                        "val_score": metrics["validation_score"],
                        "dim_score": metrics["dimensional_score"],
                        "reason": "Units incompatible in operations",
                        "errors": dim_errors,
                    }
                )

            # Category 3: Low R² (poor fit)
            elif metrics["r2_score"] < 0.90:
                self.failure_categories["poor_fit"].append(
                    {
                        "test": test_name,
                        "domain": metrics["domain"],
                        "r2": metrics["r2_score"],
                        "val_score": metrics["validation_score"],
                        "reason": f"Low R² score: {metrics['r2_score']:.4f}",
                        "engine": metrics["discovery_engine"],
                    }
                )

            # Category 4: Invalid units definition
            elif any("not defined in the unit registry" in str(e) for e in errors):
                unit_errors = [e for e in errors if "not defined" in str(e)]
                self.failure_categories["invalid_units"].append(
                    {
                        "test": test_name,
                        "domain": metrics["domain"],
                        "r2": metrics["r2_score"],
                        "val_score": metrics["validation_score"],
                        "reason": "Custom units not recognized by validator",
                        "errors": unit_errors,
                    }
                )

            # Category 5: Good R² but validation failed
            elif metrics["r2_score"] >= 0.90 and metrics["validation_score"] < 85.0:
                self.failure_categories["validation_mismatch"].append(
                    {
                        "test": test_name,
                        "domain": metrics["domain"],
                        "r2": metrics["r2_score"],
                        "val_score": metrics["validation_score"],
                        "reason": "Good fit but failed validation checks",
                        "symbolic_score": metrics["symbolic_score"],
                        "dimensional_score": metrics["dimensional_score"],
                        "errors": errors,
                    }
                )

            # Category 6: Other
            else:
                self.failure_categories["other"].append(
                    {
                        "test": test_name,
                        "domain": metrics["domain"],
                        "r2": metrics["r2_score"],
                        "val_score": metrics["validation_score"],
                        "reason": "Multiple or unclear issues",
                        "errors": errors,
                    }
                )

    def _print_summary(self):
        """Print overall summary"""
        total = len(self.results)
        failed = len(self.failures)
        passed = total - failed

        print(f"📊 OVERALL SUMMARY")
        print(f"{'=' * 80}")
        print(f"  Total tests:        {total}")
        print(f"  ✅ Passed:          {passed} ({passed / total * 100:.1f}%)")
        print(f"  ❌ Failed:          {failed} ({failed / total * 100:.1f}%)")
        print(f"  Success rate:       {passed / total * 100:.1f}%")
        print()

    def _print_r2_analysis(self):
        """Analyze R² score distribution"""
        print(f"\n📈 R² SCORE ANALYSIS")
        print(f"{'=' * 80}")

        ranges = {
            "Excellent (≥0.999)": 0,
            "Very Good (0.99-0.999)": 0,
            "Good (0.95-0.99)": 0,
            "Moderate (0.90-0.95)": 0,
            "Poor (0.80-0.90)": 0,
            "Very Poor (<0.80)": 0,
        }

        for r2 in self.r2_scores.values():
            if r2 >= 0.999:
                ranges["Excellent (≥0.999)"] += 1
            elif r2 >= 0.99:
                ranges["Very Good (0.99-0.999)"] += 1
            elif r2 >= 0.95:
                ranges["Good (0.95-0.99)"] += 1
            elif r2 >= 0.90:
                ranges["Moderate (0.90-0.95)"] += 1
            elif r2 >= 0.80:
                ranges["Poor (0.80-0.90)"] += 1
            else:
                ranges["Very Poor (<0.80)"] += 1

        for range_name, count in ranges.items():
            pct = count / len(self.r2_scores) * 100 if self.r2_scores else 0
            bar = "█" * int(pct / 5)
            print(f"  {range_name:25s} {count:3d} ({pct:5.1f}%) {bar}")

        r2_values = [r2 for r2 in self.r2_scores.values() if r2 is not None]
        if r2_values:
            print(f"\n  Mean R²:     {sum(r2_values) / len(r2_values):.4f}")
            print(f"  Median R²:   {sorted(r2_values)[len(r2_values) // 2]:.4f}")
            print(f"  Min R²:      {min(r2_values):.4f}")
            print(f"  Max R²:      {max(r2_values):.4f}")

    def _print_failure_analysis(self):
        """Detailed failure analysis"""
        print(f"\n⚠️  FAILURE ANALYSIS")
        print(f"{'=' * 80}")
        print(f"  Total failures: {len(self.failures)}")
        print()

        print(f"  Failure Categories:")
        for category, cases in self.failure_categories.items():
            if cases:
                print(
                    f"    • {category.replace('_', ' ').title():30s} {len(cases):2d} cases"
                )
        print()

        for category, cases in self.failure_categories.items():
            if not cases:
                continue

            print(f"\n  {category.replace('_', ' ').upper()}")
            print(f"  {'-' * 78}")

            for i, case in enumerate(cases, 1):
                print(f"\n  {i}. {case['test']} ({case['domain']})")
                print(f"     R²: {case.get('r2', 0):.4f}", end="")
                if "val_score" in case:
                    print(f" | Val Score: {case['val_score']:.1f}", end="")
                if "engine" in case:
                    print(f" | Engine: {case['engine']}", end="")
                print()
                print(f"     Reason: {case['reason']}")

                if case.get("errors"):
                    error_str = str(case["errors"][0])[:100]
                    print(f"     Error: {error_str}...")

    def _print_consistency_report(self):
        """Evaluate hybrid system consistency"""
        print(f"\n🔍 HYBRID SYSTEM CONSISTENCY REPORT")
        print(f"{'=' * 80}")

        engine_stats = defaultdict(
            lambda: {"total": 0, "passed": 0, "failed": 0, "r2_sum": 0}
        )

        for test_name, metrics in self.test_metrics.items():
            engine = metrics["discovery_engine"]
            engine_stats[engine]["total"] += 1
            engine_stats[engine]["r2_sum"] += metrics["r2_score"]

            if metrics["is_valid"]:
                engine_stats[engine]["passed"] += 1
            else:
                engine_stats[engine]["failed"] += 1

        print(f"\n  Engine Performance:")
        print(
            f"  {'Engine':<20} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Success %':<12} {'Avg R²':<10}"
        )
        print(f"  {'-' * 78}")

        for engine, stats in sorted(engine_stats.items()):
            success_pct = (
                stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            avg_r2 = stats["r2_sum"] / stats["total"] if stats["total"] > 0 else 0
            print(
                f"  {engine:<20} {stats['total']:<8} {stats['passed']:<8} {stats['failed']:<8} "
                f"{success_pct:<11.1f}% {avg_r2:<10.4f}"
            )

        print(f"\n  Domain Performance:")
        domain_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})

        for test_name, metrics in self.test_metrics.items():
            domain = metrics["domain"]
            domain_stats[domain]["total"] += 1
            if metrics["is_valid"]:
                domain_stats[domain]["passed"] += 1
            else:
                domain_stats[domain]["failed"] += 1

        print(
            f"  {'Domain':<20} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Success %':<12}"
        )
        print(f"  {'-' * 78}")

        for domain, stats in sorted(domain_stats.items()):
            success_pct = (
                stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            print(
                f"  {domain:<20} {stats['total']:<8} {stats['passed']:<8} {stats['failed']:<8} "
                f"{success_pct:<11.1f}%"
            )

        print(f"\n  Consistency Issues:")

        high_r2_failures = [
            t for t in self.failures if self.test_metrics[t]["r2_score"] >= 0.95
        ]
        if high_r2_failures:
            print(
                f"    • {len(high_r2_failures)} tests with high R² (≥0.95) but failed validation"
            )
            for test in high_r2_failures[:5]:
                metrics = self.test_metrics[test]
                print(
                    f"      - {test}: R²={metrics['r2_score']:.4f}, Val={metrics['validation_score']:.1f}"
                )

        low_r2_passes = [
            t
            for t, m in self.test_metrics.items()
            if m["is_valid"] and m["r2_score"] < 0.95
        ]
        if low_r2_passes:
            print(
                f"    • {len(low_r2_passes)} tests with low R² (<0.95) but passed validation"
            )

    def _print_recommendations(self):
        """Generate recommendations based on analysis"""
        print(f"\n💡 RECOMMENDATIONS")
        print(f"{'=' * 80}")

        recommendations = []

        if self.failure_categories["discovery_failure"]:
            count = len(self.failure_categories["discovery_failure"])
            recommendations.append(
                f"1. DISCOVERY ENGINE: {count} test(s) had complete discovery failures\n"
                f"   → Review variable naming (avoid reserved names like 'Q')\n"
                f"   → Check if symbolic regression parameters need tuning\n"
                f"   → Consider increasing max_complexity or iterations"
            )

        if self.failure_categories["dimensional_inconsistency"]:
            count = len(self.failure_categories["dimensional_inconsistency"])
            recommendations.append(
                f"2. DIMENSIONAL VALIDATION: {count} test(s) had dimensional inconsistencies\n"
                f"   → Review discovered expressions for physically impossible operations\n"
                f"   → Check if discovered expression adds/subtracts incompatible units\n"
                f"   → May indicate overfitting or spurious correlations"
            )

        if self.failure_categories["invalid_units"]:
            count = len(self.failure_categories["invalid_units"])
            recommendations.append(
                f"3. UNIT REGISTRY: {count} test(s) used undefined custom units\n"
                f"   → Add custom units to Pint unit registry\n"
                f"   → Or use dimensionless units for these variables\n"
                f"   → Update unit definitions in test cases"
            )

        if self.failure_categories["poor_fit"]:
            count = len(self.failure_categories["poor_fit"])
            avg_r2 = sum(c["r2"] for c in self.failure_categories["poor_fit"]) / count
            recommendations.append(
                f"4. POOR FIT: {count} test(s) had low R² scores (avg: {avg_r2:.4f})\n"
                f"   → May indicate insufficient model complexity\n"
                f"   → Check if ground truth expression is too complex for SR\n"
                f"   → Consider physics-aware constraints or domain-specific operators"
            )

        if self.failure_categories["validation_mismatch"]:
            count = len(self.failure_categories["validation_mismatch"])
            recommendations.append(
                f"5. VALIDATION MISMATCH: {count} test(s) had good R² but failed validation\n"
                f"   → Discovered expressions may be mathematically equivalent but differ in form\n"
                f"   → Validation thresholds may be too strict\n"
                f"   → Review symbolic simplification and canonical form detection"
            )

        if recommendations:
            for rec in recommendations:
                print(f"\n  {rec}")
        else:
            print(f"\n  ✅ No major issues detected. System performing well!")

        print(f"\n  OVERALL ASSESSMENT:")
        success_rate = (
            (len(self.results) - len(self.failures)) / len(self.results) * 100
        )

        if success_rate >= 90:
            assessment = "EXCELLENT - System is highly consistent and reliable"
        elif success_rate >= 75:
            assessment = "GOOD - System performs well with minor issues"
        elif success_rate >= 60:
            assessment = "MODERATE - Several consistency issues need attention"
        else:
            assessment = "NEEDS IMPROVEMENT - Significant issues affecting reliability"

        print(f"  {assessment}")
        print(f"  Success rate: {success_rate:.1f}%")
        print()


def read_file_list(file_path: str) -> List[str]:
    """Read a list of JSON file paths from a text file"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File list not found: {file_path}")

    json_files = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            json_path = Path(line)
            if not json_path.exists():
                print(f"⚠️  Warning: Line {line_num}: File not found: {line}")
                continue

            json_files.append(str(json_path))

    if not json_files:
        raise ValueError(f"No valid JSON files found in: {file_path}")

    return json_files


def compare_multiple_runs(json_files: List[str]):
    """Create comparison table from multiple JSON result files"""
    print(f"\n{'=' * 120}")
    print(f"COMPARISON TABLE - MULTIPLE TEST RUNS")
    print(f"{'=' * 120}\n")

    all_data = []

    for json_file in json_files:
        if not Path(json_file).exists():
            print(f"⚠️  Skipping {json_file} - file not found")
            continue

        try:
            analyzer = ResultsAnalyzer(json_file)
            analyzer._extract_metrics()
            analyzer._categorize_failures()

            total = len(analyzer.results)
            failed = len(analyzer.failures)
            passed = total - failed

            r2_values = [r2 for r2 in analyzer.r2_scores.values() if r2 is not None]
            avg_r2 = sum(r2_values) / len(r2_values) if r2_values else 0
            median_r2 = sorted(r2_values)[len(r2_values) // 2] if r2_values else 0

            discovery_fails = len(analyzer.failure_categories["discovery_failure"])
            dimensional_fails = len(
                analyzer.failure_categories["dimensional_inconsistency"]
            )
            poor_fit_fails = len(analyzer.failure_categories["poor_fit"])
            invalid_units_fails = len(analyzer.failure_categories["invalid_units"])
            validation_mismatch = len(
                analyzer.failure_categories["validation_mismatch"]
            )

            all_data.append(
                {
                    "file": Path(json_file).name,
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "success_rate": passed / total * 100 if total > 0 else 0,
                    "avg_r2": avg_r2,
                    "median_r2": median_r2,
                    "discovery_fails": discovery_fails,
                    "dimensional_fails": dimensional_fails,
                    "poor_fit_fails": poor_fit_fails,
                    "invalid_units_fails": invalid_units_fails,
                    "validation_mismatch": validation_mismatch,
                }
            )
        except Exception as e:
            print(f"⚠️  Error processing {json_file}: {e}")
            continue

    if not all_data:
        print("❌ No valid JSON files found")
        return

    print(
        f"{'File':<50} {'Total':<7} {'Pass':<6} {'Fail':<6} {'Success':<9} {'Avg R²':<9} {'Med R²':<9}"
    )
    print(f"{'-' * 120}")

    for data in all_data:
        print(
            f"{data['file']:<50} {data['total']:<7} {data['passed']:<6} {data['failed']:<6} "
            f"{data['success_rate']:>7.1f}% {data['avg_r2']:>8.4f} {data['median_r2']:>8.4f}"
        )

    print(f"\n{'File':<50} {'Disc':<6} {'Dim':<6} {'Fit':<6} {'Units':<6} {'ValMM':<6}")
    print(f"{'-' * 120}")
    print(f"{'':>50} {'Fail':<6} {'Fail':<6} {'Fail':<6} {'Fail':<6} {'Fail':<6}")
    print(f"{'-' * 120}")

    for data in all_data:
        print(
            f"{data['file']:<50} {data['discovery_fails']:<6} {data['dimensional_fails']:<6} "
            f"{data['poor_fit_fails']:<6} {data['invalid_units_fails']:<6} {data['validation_mismatch']:<6}"
        )

    print(f"\n{'Summary Statistics':<50}")
    print(f"{'-' * 120}")

    avg_success = sum(d["success_rate"] for d in all_data) / len(all_data)
    avg_avg_r2 = sum(d["avg_r2"] for d in all_data) / len(all_data)
    best_run = max(all_data, key=lambda x: x["success_rate"])
    worst_run = min(all_data, key=lambda x: x["success_rate"])

    print(f"  Average success rate:    {avg_success:.1f}%")
    print(f"  Average R²:              {avg_avg_r2:.4f}")
    print(
        f"  Best run:                {best_run['file']} ({best_run['success_rate']:.1f}%)"
    )
    print(
        f"  Worst run:               {worst_run['file']} ({worst_run['success_rate']:.1f}%)"
    )
    print(f"  Total runs analyzed:     {len(all_data)}")

    print(f"\n{'=' * 120}\n")


def extract_timestamp_from_filename(filename: str) -> datetime:
    """Extract timestamp from filename like 'results_20260102_064124.json'"""
    patterns = [
        r"(\d{8})_(\d{6})",
        r"(\d{14})",
        r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            if len(match.groups()) == 2:
                date_str = match.group(1) + match.group(2)
            else:
                date_str = match.group(1)

            for fmt in ["%Y%m%d%H%M%S", "%Y%m%d_%H%M%S"]:
                try:
                    date_str_clean = date_str.replace("-", "").replace("_", "")
                    return datetime.strptime(date_str_clean, fmt)
                except ValueError:
                    continue

    return None


def plot_failure_trends(directory: str, output_file: str = None):
    """Plot failure trends across multiple trial runs"""
    if not PLOTTING_AVAILABLE:
        print("❌ Plotting not available. Install matplotlib:")
        print("   pip install matplotlib")
        sys.exit(1)

    directory = Path(directory)
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        sys.exit(1)

    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in: {directory}")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f"PLOTTING FAILURE TRENDS FROM: {directory}")
    print(f"Found {len(json_files)} JSON files")
    print(f"{'=' * 80}\n")

    trial_data = []
    test_failure_tracking = defaultdict(list)

    for trial_num, json_file in enumerate(json_files, 1):
        try:
            analyzer = ResultsAnalyzer(str(json_file))
            analyzer._extract_metrics()
            analyzer._categorize_failures()

            timestamp = extract_timestamp_from_filename(json_file.name)
            if timestamp is None:
                timestamp = datetime.fromtimestamp(json_file.stat().st_mtime)

            total_tests = len(analyzer.results)
            failed_tests = len(analyzer.failures)

            discovery_fails = len(analyzer.failure_categories["discovery_failure"])
            dimensional_fails = len(
                analyzer.failure_categories["dimensional_inconsistency"]
            )
            poor_fit_fails = len(analyzer.failure_categories["poor_fit"])
            invalid_units_fails = len(analyzer.failure_categories["invalid_units"])
            validation_mismatch = len(
                analyzer.failure_categories["validation_mismatch"]
            )

            trial_data.append(
                {
                    "trial": trial_num,
                    "timestamp": timestamp,
                    "filename": json_file.name,
                    "total": total_tests,
                    "failed": failed_tests,
                    "passed": total_tests - failed_tests,
                    "discovery_fails": discovery_fails,
                    "dimensional_fails": dimensional_fails,
                    "poor_fit_fails": poor_fit_fails,
                    "invalid_units_fails": invalid_units_fails,
                    "validation_mismatch": validation_mismatch,
                }
            )

            for failed_test in analyzer.failures:
                test_failure_tracking[failed_test].append(trial_num)

            print(f"  ✓ Processed trial {trial_num}: {json_file.name}")

        except Exception as e:
            print(f"  ⚠️  Error processing {json_file.name}: {e}")
            continue

    if not trial_data:
        print("❌ No valid trial data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "HypatiaX Test Results - Failure Trends", fontsize=16, fontweight="bold"
    )

    trials = [d["trial"] for d in trial_data]

    # Plot 1: Overall pass/fail trend
    ax1 = axes[0, 0]
    passed = [d["passed"] for d in trial_data]
    failed = [d["failed"] for d in trial_data]
    total = [d["total"] for d in trial_data]

    ax1.plot(
        trials, passed, "o-", color="green", linewidth=2, markersize=8, label="Passed"
    )
    ax1.plot(
        trials, failed, "o-", color="red", linewidth=2, markersize=8, label="Failed"
    )
    ax1.set_xlabel("Trial Number", fontsize=12)
    ax1.set_ylabel("Number of Tests", fontsize=12)
    ax1.set_title("Pass/Fail Trend Over Trials", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Failure categories stacked area
    ax2 = axes[0, 1]
    discovery = [d["discovery_fails"] for d in trial_data]
    dimensional = [d["dimensional_fails"] for d in trial_data]
    poor_fit = [d["poor_fit_fails"] for d in trial_data]
    invalid_units = [d["invalid_units_fails"] for d in trial_data]
    val_mismatch = [d["validation_mismatch"] for d in trial_data]
    ax2.stackplot(
        trials,
        discovery,
        dimensional,
        poor_fit,
        invalid_units,
        val_mismatch,
        labels=[
            "Discovery",
            "Dimensional",
            "Poor Fit",
            "Invalid Units",
            "Validation Mismatch",
        ],
        colors=["#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24", "#a29bfe"],
        alpha=0.8,
    )
    ax2.set_xlabel("Trial Number", fontsize=12)
    ax2.set_ylabel("Number of Failures", fontsize=12)
    ax2.set_title("Failure Categories (Stacked)", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Success rate over time
    ax3 = axes[1, 0]
    success_rates = [(d["passed"] / d["total"] * 100) for d in trial_data]

    ax3.plot(trials, success_rates, "o-", color="purple", linewidth=2, markersize=8)
    ax3.axhline(
        y=90,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="90% Target",
    )
    ax3.axhline(
        y=75,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="75% Threshold",
    )
    ax3.set_xlabel("Trial Number", fontsize=12)
    ax3.set_ylabel("Success Rate (%)", fontsize=12)
    ax3.set_title("Success Rate Over Trials", fontsize=14, fontweight="bold")
    ax3.set_ylim([0, 105])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    mean_success = sum(success_rates) / len(success_rates)
    ax3.text(
        0.02,
        0.98,
        f"Mean: {mean_success:.1f}%",
        transform=ax3.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Plot 4: Consistently failing tests
    ax4 = axes[1, 1]

    persistent_failures = {
        test: trials_list
        for test, trials_list in test_failure_tracking.items()
        if len(trials_list) >= 2
    }

    if persistent_failures:
        sorted_failures = sorted(
            persistent_failures.items(), key=lambda x: len(x[1]), reverse=True
        )[:15]

        test_names = [test[:30] for test, _ in sorted_failures]
        failure_counts = [len(trials_list) for _, trials_list in sorted_failures]

        bars = ax4.barh(
            range(len(test_names)), failure_counts, color="crimson", alpha=0.7
        )
        ax4.set_yticks(range(len(test_names)))
        ax4.set_yticklabels(test_names, fontsize=9)
        ax4.set_xlabel("Number of Failed Trials", fontsize=12)
        ax4.set_title("Most Frequently Failing Tests", fontsize=14, fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="x")

        for i, (bar, count) in enumerate(zip(bars, failure_counts)):
            ax4.text(count, i, f" {count}", va="center", fontsize=9)
    else:
        ax4.text(
            0.5,
            0.5,
            "No persistent failures detected",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax4.transAxes,
        )
        ax4.set_title("Most Frequently Failing Tests", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"\n✅ Plot saved to: {output_file}")
    else:
        output_file = directory / "failure_trends.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"\n✅ Plot saved to: {output_file}")

    plt.close()

    print(f"\n{'=' * 80}")
    print(f"TREND ANALYSIS SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total trials analyzed:       {len(trial_data)}")
    print(f"  Average success rate:        {mean_success:.1f}%")
    print(
        f"  Best trial:                  {max(trial_data, key=lambda x: x['passed'] / x['total'])['filename']}"
    )
    print(
        f"  Worst trial:                 {min(trial_data, key=lambda x: x['passed'] / x['total'])['filename']}"
    )
    print(f"  Tests with persistent fails: {len(persistent_failures)}")
    print(f"{'=' * 80}\n")


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Analyze HypatiaX test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single results file
  python analyze_results.py results.json

  # Compare multiple runs
  python analyze_results.py --compare run1.json run2.json run3.json

  # Process file list
  python analyze_results.py --file-list json_files.txt

  # Plot failure trends
  python analyze_results.py --plot-failures results_directory/
  python analyze_results.py --plot-failures results/ --output trends.png
        """,
    )

    parser.add_argument(
        "json_file", nargs="?", help="Single JSON results file to analyze"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="FILE",
        help="Compare multiple JSON result files",
    )
    parser.add_argument(
        "--file-list",
        metavar="FILE",
        help="Text file containing list of JSON files (one per line)",
    )
    parser.add_argument(
        "--plot-failures",
        metavar="DIR",
        help="Plot failure trends from directory of JSON files",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output file for plot (default: failure_trends.png)",
    )

    args = parser.parse_args()

    if args.plot_failures:
        plot_failure_trends(args.plot_failures, args.output)
        return

    if args.compare:
        compare_multiple_runs(args.compare)
        return

    if args.file_list:
        json_files = read_file_list(args.file_list)
        print(f"\n📋 Processing {len(json_files)} files from: {args.file_list}\n")
        compare_multiple_runs(json_files)
        return

    if args.json_file:
        try:
            analyzer = ResultsAnalyzer(args.json_file)
            analyzer.analyze_all()
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
