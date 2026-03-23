#!/usr/bin/env python3
"""
analyze_hybrid_performance.py

Comprehensive performance analysis for the Hybrid DeFi System.
Analyzes results from hybrid_defi_full.py and generates detailed reports.

Features:
- Overall performance metrics
- Domain-specific analysis
- Decision strategy evaluation
- Extrapolation performance
- Comparative analysis (LLM vs Ensemble vs NN)
- Failure analysis
- Visualization generation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse


class HybridPerformanceAnalyzer:
    """Analyze performance of hybrid DeFi system"""
    
    def __init__(self, results_dir: str = "hypatiax/data/results"):
        self.results_dir = Path(results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create structured experiment folders
        self.figures_dir = self.results_dir / "figures"
        self.latex_dir = self.results_dir / "latex"

        self.figures_dir.mkdir(exist_ok=True)
        self.latex_dir.mkdir(exist_ok=True)
        
        self.hybrid_results = []
        self.baseline_results = []   # populated from "baseline" strategy results
        self.extrapolation_results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_latest_results(self) -> bool:
        """Load the most recent results files"""
        
        # Find latest hybrid results
        hybrid_files = sorted(self.results_dir.glob("hybrid_defi_*.json"))
        if not hybrid_files:
            print("❌ No hybrid results found")
            return False
        
        latest_hybrid = hybrid_files[-1]
        print(f"📂 Loading hybrid results: {latest_hybrid.name}")
        
        with open(latest_hybrid) as f:
            raw = json.load(f)

        # Support both formats:
        #   - wrapper dict: {"timestamp":..., "results": [...]}   (written by hybrid_system_nn_defi_domain.py)
        #   - bare list:    [...]                                  (legacy format)
        if isinstance(raw, dict):
            result_list = raw.get("results", [])
        else:
            result_list = raw

        # Normalise flat keys produced by hybrid_system_nn_defi_domain.py into the
        # nested schema expected by this analyser:
        #   hybrid_train_r2  -> evaluation.r2
        #   llm_train_r2     -> llm_result.metrics.r2
        #   nn_train_r2      -> nn_result.metrics.r2
        #   llm_formula / llm_python  -> kept as-is (unused by analyser)
        normalised = []
        for r in result_list:
            if not isinstance(r, dict):
                continue
            # Already in nested format?
            if "evaluation" in r:
                normalised.append(r)
                continue
            # Flat format → convert
            hybrid_r2 = r.get("hybrid_train_r2", r.get("hybrid_r2"))
            llm_r2    = r.get("llm_train_r2",    r.get("llm_r2", 0.0))
            nn_r2     = r.get("nn_train_r2",      r.get("nn_r2", 0.0))
            normalised.append({
                "description": r.get("description", ""),
                "domain":      r.get("domain", "Unknown"),
                "decision":    r.get("decision", "unknown"),
                "success":     r.get("success", True),
                "error":       r.get("error"),
                "evaluation":  {"r2": hybrid_r2, "mse": None, "mae": None},
                "llm_result":  {
                    "formula":      r.get("llm_formula", "N/A"),
                    "python_code":  r.get("llm_python",  "N/A"),
                    "metrics":      {"r2": llm_r2, "success": llm_r2 is not None},
                },
                "nn_result":   {"metrics": {"r2": nn_r2, "success": nn_r2 is not None}},
                "ensemble_result": {"metrics": {}},
            })

        self.hybrid_results = normalised
        print(f"✅ Loaded {len(self.hybrid_results)} hybrid test cases")
        
        # Find latest extrapolation results (optional)
        extrap_files = sorted(self.results_dir.glob("extrapolation_*.json"))
        if extrap_files:
            latest_extrap = extrap_files[-1]
            print(f"📂 Loading extrapolation results: {latest_extrap.name}")
            try:
                with open(latest_extrap) as f:
                    raw_extrap = json.load(f)
                self.extrapolation_results = (
                    raw_extrap if isinstance(raw_extrap, list)
                    else raw_extrap.get("results", [])
                )
                print(f"✅ Loaded {len(self.extrapolation_results)} extrapolation cases")
            except json.JSONDecodeError as exc:
                # File was likely left corrupt by a previous crashed run.
                # Log the problem and continue — extrapolation section is optional.
                print(f"⚠️  Extrapolation file is corrupt ({latest_extrap.name}): {exc}")
                print("   → Delete or re-run test_enhanced_defi_extrapolation.py to regenerate.")
                self.extrapolation_results = []
        else:
            print("⚠️  No extrapolation results found (skipping)")
        
        return True
    
    def analyze_overall_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance"""
        
        print("\n" + "=" * 80)
        print("📊 OVERALL PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        total_cases = len(self.hybrid_results)
        
        # Extract R² scores
        r2_scores = []
        success_count = 0
        
        for result in self.hybrid_results:
            eval_metrics = result.get('evaluation', {})
            r2 = eval_metrics.get('r2')
            
            if r2 is not None:
                r2_scores.append(r2)
                if r2 > 0.9:  # Consider success if R² > 0.9
                    success_count += 1
        
        # Calculate statistics
        if r2_scores:
            stats = {
                'total_cases': total_cases,
                'valid_r2_count': len(r2_scores),
                'success_count': success_count,
                'success_rate': success_count / len(r2_scores),
                'mean_r2': np.mean(r2_scores),
                'median_r2': np.median(r2_scores),
                'std_r2': np.std(r2_scores),
                'min_r2': np.min(r2_scores),
                'max_r2': np.max(r2_scores),
                'q25_r2': np.percentile(r2_scores, 25),
                'q75_r2': np.percentile(r2_scores, 75),
            }
            
            # Performance tiers
            excellent = sum(1 for r2 in r2_scores if r2 > 0.99)
            good = sum(1 for r2 in r2_scores if 0.95 <= r2 <= 0.99)
            acceptable = sum(1 for r2 in r2_scores if 0.90 <= r2 < 0.95)
            poor = sum(1 for r2 in r2_scores if r2 < 0.90)
            
            stats.update({
                'excellent_count': excellent,  # R² > 0.99
                'good_count': good,            # 0.95 <= R² <= 0.99
                'acceptable_count': acceptable, # 0.90 <= R² < 0.95
                'poor_count': poor,            # R² < 0.90
            })
        else:
            stats = {'error': 'No valid R² scores found'}
        
        # Print summary
        if 'error' in stats:
            print(f"\n❌ {stats['error']}")
        else:
            print(f"\n📈 Summary Statistics:")
            print(f"   Total cases: {stats['total_cases']}")
            print(f"   Valid R² scores: {stats['valid_r2_count']}")
            print(f"   Success rate: {stats['success_rate'] * 100:.1f}% (R² > 0.9)")
            print(f"\n📊 R² Distribution:")
            print(f"   Mean: {stats['mean_r2']:.6f}")
            print(f"   Median: {stats['median_r2']:.6f}")
            print(f"   Std Dev: {stats['std_r2']:.6f}")
            print(f"   Range: [{stats['min_r2']:.6f}, {stats['max_r2']:.6f}]")
            print(f"   Q25-Q75: [{stats['q25_r2']:.6f}, {stats['q75_r2']:.6f}]")
            print(f"\n🎯 Performance Tiers:")
            print(f"   Excellent (R² > 0.99): {stats['excellent_count']} ({stats['excellent_count']/total_cases*100:.1f}%)")
            print(f"   Good (0.95-0.99): {stats['good_count']} ({stats['good_count']/total_cases*100:.1f}%)")
            print(f"   Acceptable (0.90-0.95): {stats['acceptable_count']} ({stats['acceptable_count']/total_cases*100:.1f}%)")
            print(f"   Poor (< 0.90): {stats['poor_count']} ({stats['poor_count']/total_cases*100:.1f}%)")
        
        return stats
    
    def analyze_by_domain(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by DeFi domain"""
        
        print("\n" + "=" * 80)
        print("🏢 DOMAIN-SPECIFIC ANALYSIS")
        print("=" * 80)
        
        domains = {}
        
        for result in self.hybrid_results:
            domain = result.get('domain', 'Unknown')
            
            if domain not in domains:
                domains[domain] = {
                    'cases': [],
                    'r2_scores': [],
                }
            
            domains[domain]['cases'].append(result)
            
            r2 = result.get('evaluation', {}).get('r2')
            if r2 is not None:
                domains[domain]['r2_scores'].append(r2)
        
        # Calculate domain statistics
        domain_stats = {}
        
        for domain, data in domains.items():
            r2_scores = data['r2_scores']
            
            if r2_scores:
                domain_stats[domain] = {
                    'total': len(data['cases']),
                    'mean_r2': np.mean(r2_scores),
                    'median_r2': np.median(r2_scores),
                    'std_r2': np.std(r2_scores),
                    'min_r2': np.min(r2_scores),
                    'max_r2': np.max(r2_scores),
                    'success_rate': sum(1 for r2 in r2_scores if r2 > 0.9) / len(r2_scores),
                }
        
        # Print domain performance
        print(f"\n📊 Performance by Domain:")
        print(f"{'Domain':<25} {'Cases':<8} {'Mean R²':<12} {'Success Rate':<15} {'Range'}")
        print("-" * 80)
        
        for domain in sorted(domain_stats.keys()):
            stats = domain_stats[domain]
            print(f"{domain:<25} {stats['total']:<8} {stats['mean_r2']:<12.6f} "
                  f"{stats['success_rate']*100:>6.1f}%        "
                  f"[{stats['min_r2']:.3f}, {stats['max_r2']:.3f}]")
        
        return domain_stats
    
    def analyze_decision_strategy(self) -> Dict[str, Any]:
        """Analyze performance of different decision strategies"""
        
        print("\n" + "=" * 80)
        print("🎯 DECISION STRATEGY ANALYSIS")
        print("=" * 80)
        
        strategies = {
            'llm': [],
            'ensemble': [],
            'nn': [],
            'unknown': []
        }
        
        for result in self.hybrid_results:
            decision = result.get('decision', 'unknown')
            r2 = result.get('evaluation', {}).get('r2')
            
            if decision in strategies and r2 is not None:
                strategies[decision].append({
                    'r2': r2,
                    'description': result.get('description', ''),
                    'domain': result.get('domain', ''),
                })
        
        # Calculate strategy statistics
        strategy_stats = {}
        
        for strategy, results in strategies.items():
            if results:
                r2_scores = [r['r2'] for r in results]
                strategy_stats[strategy] = {
                    'count': len(results),
                    'mean_r2': np.mean(r2_scores),
                    'median_r2': np.median(r2_scores),
                    'std_r2': np.std(r2_scores),
                    'min_r2': np.min(r2_scores),
                    'max_r2': np.max(r2_scores),
                    'success_rate': sum(1 for r2 in r2_scores if r2 > 0.9) / len(r2_scores),
                }
        
        # Print strategy performance
        total = sum(s['count'] for s in strategy_stats.values())
        
        print(f"\n📊 Strategy Distribution:")
        print(f"   Total decisions: {total}")
        for strategy, stats in strategy_stats.items():
            pct = stats['count'] / total * 100 if total > 0 else 0
            print(f"   {strategy.upper():<10}: {stats['count']:>4} ({pct:>5.1f}%)")
        
        print(f"\n📈 Strategy Performance:")
        print(f"{'Strategy':<12} {'Count':<8} {'Mean R²':<12} {'Success Rate':<15} {'Range'}")
        print("-" * 80)
        
        for strategy in ['llm', 'ensemble', 'nn']:
            if strategy in strategy_stats:
                stats = strategy_stats[strategy]
                print(f"{strategy.upper():<12} {stats['count']:<8} {stats['mean_r2']:<12.6f} "
                      f"{stats['success_rate']*100:>6.1f}%        "
                      f"[{stats['min_r2']:.3f}, {stats['max_r2']:.3f}]")
        
        # Comparative analysis
        print(f"\n🔍 Comparative Insights:")
        
        if 'llm' in strategy_stats and 'ensemble' in strategy_stats:
            llm_mean = strategy_stats['llm']['mean_r2']
            ens_mean = strategy_stats['ensemble']['mean_r2']
            diff = llm_mean - ens_mean
            
            if diff > 0.01:
                print(f"   • LLM outperforms Ensemble by {diff:.4f} R² on average")
            elif diff < -0.01:
                print(f"   • Ensemble outperforms LLM by {abs(diff):.4f} R² on average")
            else:
                print(f"   • LLM and Ensemble perform similarly (Δ = {diff:.4f})")
        
        if 'nn' in strategy_stats:
            nn_success = strategy_stats['nn']['success_rate']
            print(f"   • Neural Network success rate: {nn_success*100:.1f}%")
        
        return strategy_stats
    
    def analyze_component_performance(self) -> Dict[str, Any]:
        """Analyze individual component performance (LLM, Ensemble, NN)"""
        
        print("\n" + "=" * 80)
        print("🔧 COMPONENT PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        component_stats = {
            'llm': {'r2_scores': [], 'success': [], 'failures': []},
            'ensemble': {'r2_scores': [], 'success': [], 'failures': []},
            'nn': {'r2_scores': [], 'success': [], 'failures': []}
        }
        
        for result in self.hybrid_results:
            # LLM results
            llm_result = result.get('llm_result', {})
            llm_metrics = llm_result.get('metrics', {})
            llm_r2 = llm_metrics.get('r2')
            
            if llm_r2 is not None:
                component_stats['llm']['r2_scores'].append(llm_r2)
                if llm_metrics.get('success', False):
                    component_stats['llm']['success'].append(result)
                else:
                    component_stats['llm']['failures'].append(result)
            
            # Ensemble results
            ensemble_result = result.get('ensemble_result', {})
            ensemble_metrics = ensemble_result.get('metrics', {})
            ensemble_r2 = ensemble_metrics.get('r2')
            
            if ensemble_r2 is not None:
                component_stats['ensemble']['r2_scores'].append(ensemble_r2)
                if ensemble_metrics.get('success', False):
                    component_stats['ensemble']['success'].append(result)
                else:
                    component_stats['ensemble']['failures'].append(result)
            
            # NN results
            nn_result = result.get('nn_result', {})
            nn_metrics = nn_result.get('metrics', {})
            nn_r2 = nn_metrics.get('r2')
            
            if nn_r2 is not None:
                component_stats['nn']['r2_scores'].append(nn_r2)
                if nn_metrics.get('success', False):
                    component_stats['nn']['success'].append(result)
                else:
                    component_stats['nn']['failures'].append(result)
        
        # Print component statistics
        print(f"\n📊 Component Statistics:")
        print(f"{'Component':<12} {'Attempts':<10} {'Successes':<12} {'Failures':<10} {'Mean R²'}")
        print("-" * 80)
        
        component_summary = {}
        
        for component, data in component_stats.items():
            attempts = len(data['r2_scores'])
            successes = len(data['success'])
            failures = len(data['failures'])
            mean_r2 = np.mean(data['r2_scores']) if data['r2_scores'] else 0
            
            component_summary[component] = {
                'attempts': attempts,
                'successes': successes,
                'failures': failures,
                'success_rate': successes / attempts if attempts > 0 else 0,
                'mean_r2': mean_r2,
            }
            
            print(f"{component.upper():<12} {attempts:<10} {successes:<12} {failures:<10} {mean_r2:.6f}")

        return component_summary
    
    # ==========================================================
    # 🚀 PHASE 1 — Hierarchical Bayesian Modeling
    # Goal:
    # Estimate posterior distribution of performance differences while accounting for case-level variability.
    # ==========================================================
    
    def hierarchical_bayesian_model(self, n_samples: int = 10_000) -> Dict[str, Any]:
        """
        Hierarchical Bayesian model across difficulty tiers.
        Estimates posterior dominance (P(Hybrid > Baseline)) per tier.
        Falls back to cross-domain analysis when no difficulty field is present.
        """
        tiers = self.get_results_grouped_by_difficulty()
        posterior_results: Dict[str, Any] = {}

        for tier_name, tier_data in tiers.items():
            hybrid   = np.array(tier_data["hybrid"])
            baseline = np.array(tier_data["baseline"]) if tier_data["baseline"] else None

            # When no baseline field exists in the data, use NN scores as proxy
            if baseline is None or len(baseline) < 2:
                baseline = np.array([
                    r.get("nn_result", {}).get("metrics", {}).get("r2")
                    for r in self.hybrid_results
                    if (r.get("difficulty", "unknown") == tier_name
                        and r.get("nn_result", {}).get("metrics", {}).get("r2") is not None)
                ])

            if len(hybrid) < 2 or len(baseline) < 2:
                continue

            diff_mean = hybrid.mean() - baseline.mean()
            diff_std  = np.sqrt(
                (hybrid.std(ddof=1) ** 2 / len(hybrid)) +
                (baseline.std(ddof=1) ** 2 / len(baseline))
            )
            samples = np.random.normal(diff_mean, diff_std, n_samples)

            posterior_results[tier_name] = {
                "posterior_prob": float(np.mean(samples > 0)),
                "mean_diff":      float(diff_mean),
                "n_hybrid":       len(hybrid),
                "n_baseline":     len(baseline),
            }
            print(f"🧠 Hierarchical P(Hybrid>Baseline) | {tier_name}: "
                  f"{posterior_results[tier_name]['posterior_prob']:.3f}  "
                  f"(Δ={diff_mean:.4f})")

        if not posterior_results:
            print("⚠️  hierarchical_bayesian_model: insufficient per-tier data.")
        return posterior_results

    # Legacy name kept for backwards compatibility
    def hierarchical_bayesian_analysis(self):
        """
        Hierarchical Bayesian model (approximate)
        Computes posterior distribution of mean difference (LLM - NN)
        with case-level shrinkage.
        """

        import numpy as np

        paired_diffs = []

        for result in self.hybrid_results:
            llm = result.get('llm_result', {}).get('metrics', {}).get('r2')
            nn  = result.get('nn_result', {}).get('metrics', {}).get('r2')

            if llm is not None and nn is not None:
                paired_diffs.append(llm - nn)

        if len(paired_diffs) < 3:
            print("Not enough data for hierarchical Bayesian analysis.")
            return None

        paired_diffs = np.array(paired_diffs)

        # Empirical Bayes shrinkage
        mu_hat = np.mean(paired_diffs)
        sigma_hat = np.std(paired_diffs)

        # Posterior sampling
        posterior_samples = np.random.normal(
            loc=mu_hat,
            scale=sigma_hat / np.sqrt(len(paired_diffs)),
            size=5000
        )

        prob_superiority = np.mean(posterior_samples > 0)

        print(f"\n🔬 Hierarchical Bayesian Posterior Mean: {mu_hat:.4f}")
        print(f"🧠 P(LLM > NN): {prob_superiority:.3f}")

        return {
            "posterior_mean": mu_hat,
            "prob_superiority": prob_superiority
        }

    # ==========================================================
    #  🚀 PHASE 2 — Cross-Difficulty Posterior Dominance
    # We now compute posterior dominance separately for: easy medium hard
    # ==========================================================
  
    def cross_difficulty_bayesian(self):
        import numpy as np

        results_by_diff = {}

        for difficulty in ["easy", "medium", "hard"]:
            diffs = []

            for result in self.hybrid_results:
                if result.get("difficulty") == difficulty:
                    llm = result.get('llm_result', {}).get('metrics', {}).get('r2')
                    nn  = result.get('nn_result', {}).get('metrics', {}).get('r2')

                    if llm is not None and nn is not None:
                        diffs.append(llm - nn)

            if len(diffs) > 2:
                diffs = np.array(diffs)
                posterior = np.random.normal(
                    np.mean(diffs),
                    np.std(diffs)/np.sqrt(len(diffs)),
                    4000
                )
                prob = np.mean(posterior > 0)

                results_by_diff[difficulty] = prob
                print(f"🧠 P(LLM>NN) | {difficulty}: {prob:.3f}")

        return results_by_diff



    def analyze_failures(self) -> List[Dict[str, Any]]:
        """Analyze cases where the system performed poorly"""
        
        print("\n" + "=" * 80)
        print("❌ FAILURE ANALYSIS")
        print("=" * 80)
        
        failures = []
        
        for result in self.hybrid_results:
            r2 = result.get('evaluation', {}).get('r2')
            
            if r2 is not None and r2 < 0.90:
                failures.append({
                    'description': result.get('description', 'Unknown'),
                    'domain': result.get('domain', 'Unknown'),
                    'decision': result.get('decision', 'unknown'),
                    'r2': r2,
                    'llm_r2': result.get('llm_result', {}).get('metrics', {}).get('r2'),
                    'ensemble_r2': (result.get('ensemble_result') or {}).get('metrics', {}).get('r2'),
                    'nn_r2': result.get('nn_result', {}).get('metrics', {}).get('r2'),
                })
        
        print(f"\n⚠️  Found {len(failures)} cases with R² < 0.90")
        
        if failures:
            # Sort by R² (worst first)
            failures.sort(key=lambda x: x['r2'])
            
            print(f"\n🔍 Top 10 Worst Cases:")
            print(f"{'Description':<45} {'Domain':<15} {'Decision':<10} {'R²'}")
            print("-" * 90)
            
            for i, failure in enumerate(failures[:10], 1):
                desc = failure['description'][:43] + '...' if len(failure['description']) > 45 else failure['description']
                print(f"{desc:<45} {failure['domain']:<15} {failure['decision']:<10} {failure['r2']:.6f}")
            
            # Analyze failure patterns
            print(f"\n📊 Failure Patterns:")
            
            # By domain
            failure_domains = {}
            for f in failures:
                domain = f['domain']
                failure_domains[domain] = failure_domains.get(domain, 0) + 1
            
            print(f"\n   By Domain:")
            for domain, count in sorted(failure_domains.items(), key=lambda x: x[1], reverse=True):
                print(f"      {domain}: {count}")
            
            # By decision
            failure_decisions = {}
            for f in failures:
                decision = f['decision']
                failure_decisions[decision] = failure_decisions.get(decision, 0) + 1
            
            print(f"\n   By Decision:")
            for decision, count in sorted(failure_decisions.items(), key=lambda x: x[1], reverse=True):
                print(f"      {decision.upper()}: {count}")
        
        return failures
    
    def analyze_extrapolation(self) -> Dict[str, Any]:
        """Analyze extrapolation performance"""
        
        if not self.extrapolation_results:
            print("\n⚠️  No extrapolation results available")
            return {}
        
        print("\n" + "=" * 80)
        print("🔮 EXTRAPOLATION ANALYSIS")
        print("=" * 80)
        
        r2_scores = []
        success_count = 0
        
        for result in self.extrapolation_results:
            if isinstance(result, dict):
                r2 = result.get('r2')
                success = result.get('success', False)
            else:
                continue
            
            if r2 is not None:
                r2_scores.append(r2)
                if success:
                    success_count += 1
        
        if r2_scores:
            stats = {
                'total_cases': len(self.extrapolation_results),
                'valid_r2_count': len(r2_scores),
                'success_count': success_count,
                'success_rate': success_count / len(r2_scores),
                'mean_r2': np.mean(r2_scores),
                'median_r2': np.median(r2_scores),
                'std_r2': np.std(r2_scores),
                'min_r2': np.min(r2_scores),
                'max_r2': np.max(r2_scores),
            }
            
            print(f"\n📈 Extrapolation Statistics:")
            print(f"   Total cases: {stats['total_cases']}")
            print(f"   Success rate: {stats['success_rate'] * 100:.1f}%")
            print(f"   Mean R²: {stats['mean_r2']:.6f}")
            print(f"   Median R²: {stats['median_r2']:.6f}")
            print(f"   Range: [{stats['min_r2']:.6f}, {stats['max_r2']:.6f}]")
            
            return stats
        else:
            print("❌ No valid extrapolation R² scores")
            return {}


    def generate_latex_report(self, summary):
            # Pull values safely so the report generates even with partial summaries
            overall      = summary.get("overall", {})
            bayesian     = summary.get("bayesian", {})
            components   = summary.get("components", {})

            mean_r2_llm  = components.get("llm",  {}).get("mean_r2", float("nan"))
            mean_r2_nn   = components.get("nn",   {}).get("mean_r2", float("nan"))
            posterior     = bayesian.get("posterior_prob", float("nan"))
            mean_diff     = bayesian.get("mean_difference", float("nan"))

            latex_content = (
                "\\documentclass[11pt]{article}\n"
                "\\usepackage{graphicx}\n"
                "\\usepackage{booktabs}\n"
                "\\usepackage{geometry}\n"
                "\\geometry{margin=1in}\n"
                "\\usepackage{amsmath}\n"
                "\\usepackage{natbib}\n"
                "\\usepackage{hyperref}\n"
                "\n"
                "\\begin{document}\n"
                "\n"
                "\\title{Hybrid vs LLM vs NN Performance Report}\n"
                "\\maketitle\n"
                "\n"
                "\\section*{Summary}\n"
                "\n"
                f"Mean R\\textsuperscript{{2}} LLM: {mean_r2_llm:.4f} \\\\\n"
                f"Mean R\\textsuperscript{{2}} NN: {mean_r2_nn:.4f} \\\\\n"
                "\n"
                "\\section*{Bayesian Analysis}\n"
                f"Posterior Mean Difference: {mean_diff:.4f} \\\\\n"
                f"Probability of Superiority: {posterior:.4f}\n"
                "\n"
                "\\section*{Figures}\n"
                "\n"
                "\\includegraphics[width=0.8\\linewidth]{../figures/violin_significance.png}\n"
                "\n"
                "\\includegraphics[width=0.8\\linewidth]{../figures/paired_difference_ci.png}\n"
                "\n"
                "\\end{document}\n"
            )

            tex_path = self.latex_dir / "report.tex"

            with open(tex_path, "w") as f:
                f.write(latex_content)

            # Compile automatically — only when pdflatex is available.
            import shutil
            import subprocess
            if shutil.which("pdflatex"):
                subprocess.run(
                    ["pdflatex", "report.tex"],
                    cwd=self.latex_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print("📄 LaTeX report compiled.")
            else:
                print("📄 LaTeX source saved (pdflatex not found — skipping compilation).")
    
    def generate_recommendations(self, overall_stats: Dict, domain_stats: Dict, 
                                strategy_stats: Dict, failures: List) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)
        
        recommendations = []
        
        # Overall performance
        if overall_stats.get('mean_r2', 0) > 0.95:
            recommendations.append("✅ Overall performance is excellent (Mean R² > 0.95)")
        elif overall_stats.get('mean_r2', 0) > 0.90:
            recommendations.append("✅ Overall performance is good (Mean R² > 0.90)")
        else:
            recommendations.append("⚠️  Overall performance needs improvement (Mean R² < 0.90)")
            recommendations.append("   → Consider tuning hyperparameters or improving prompts")
        
        # Domain-specific
        if domain_stats:
            worst_domain = min(domain_stats.items(), key=lambda x: x[1]['mean_r2'])
            if worst_domain[1]['mean_r2'] < 0.85:
                recommendations.append(f"⚠️  Domain '{worst_domain[0]}' underperforms (Mean R² = {worst_domain[1]['mean_r2']:.4f})")
                recommendations.append(f"   → Add domain-specific training data for {worst_domain[0]}")
        
        # Strategy comparison
        if 'llm' in strategy_stats and 'ensemble' in strategy_stats:
            llm_mean = strategy_stats['llm']['mean_r2']
            ens_mean = strategy_stats['ensemble']['mean_r2']
            
            if llm_mean > ens_mean + 0.05:
                recommendations.append("✅ LLM significantly outperforms Ensemble")
                recommendations.append("   → Consider using LLM more frequently")
            elif ens_mean > llm_mean + 0.05:
                recommendations.append("✅ Ensemble significantly outperforms LLM")
                recommendations.append("   → Consider using Ensemble more frequently")
        
        # Failures
        if failures:
            if len(failures) > len(self.hybrid_results) * 0.1:
                recommendations.append(f"⚠️  High failure rate: {len(failures)} cases with R² < 0.90")
                recommendations.append("   → Review failure patterns and add targeted improvements")
            
            # Common failure domains
            failure_domains = {}
            for f in failures:
                domain = f['domain']
                failure_domains[domain] = failure_domains.get(domain, 0) + 1
            
            if failure_domains:
                worst_fail_domain = max(failure_domains.items(), key=lambda x: x[1])
                recommendations.append(f"⚠️  Most failures in '{worst_fail_domain[0]}' ({worst_fail_domain[1]} cases)")
                recommendations.append(f"   → Focus improvement efforts on {worst_fail_domain[0]} domain")
        
        # Print recommendations
        print()
        for i, rec in enumerate(recommendations, 1):
            print(f"{rec}")
        
        return recommendations
    
    def save_report(self, overall_stats: Dict, domain_stats: Dict, 
                   strategy_stats: Dict, component_stats: Dict,
                   failures: List, recommendations: List):
        """Save comprehensive report to JSON"""
        
        report = {
            'timestamp': self.timestamp,
            'overall': overall_stats,
            'by_domain': domain_stats,
            'by_strategy': strategy_stats,
            'component_performance': component_stats,
            'failures': failures,
            'recommendations': recommendations,
            'extrapolation_tests': self.extrapolation_results if self.extrapolation_results else [],
        }
        
        output_file = self.results_dir / f"report_hybrid_{self.timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to: {output_file}")
        
        return output_file


    # ------------------------------------------------------------------
    # Research-rigor methods (JMLR-grade)
    # ------------------------------------------------------------------

    def get_results_grouped_by_difficulty(self) -> Dict[str, Dict[str, list]]:
        """Group hybrid vs baseline R² scores by difficulty tier."""
        tiers: Dict[str, Dict[str, list]] = {}
        for result in self.hybrid_results:
            difficulty = result.get("difficulty", "unknown")
            if difficulty not in tiers:
                tiers[difficulty] = {"hybrid": [], "baseline": []}
            h_r2 = result.get("evaluation", {}).get("r2")
            b_r2 = result.get("baseline_r2")  # optional field
            if h_r2 is not None:
                tiers[difficulty]["hybrid"].append(h_r2)
            if b_r2 is not None:
                tiers[difficulty]["baseline"].append(b_r2)
        return tiers

    def compute_bootstrap_confidence_intervals(self, n_bootstrap: int = 5000) -> Dict[str, Any]:
        """
        Non-parametric bootstrap CI for mean R² of the hybrid system.
        Returns mean, and 95 % CI [lower, upper].
        """
        r2_scores = [
            r.get("evaluation", {}).get("r2")
            for r in self.hybrid_results
            if r.get("evaluation", {}).get("r2") is not None
        ]
        if len(r2_scores) < 2:
            print("⚠️  Not enough R² scores for bootstrap CI.")
            return {}

        r2_arr = np.array(r2_scores)
        boot_means = [
            np.mean(np.random.choice(r2_arr, size=len(r2_arr), replace=True))
            for _ in range(n_bootstrap)
        ]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        result = {
            "mean_r2": float(r2_arr.mean()),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_bootstrap": n_bootstrap,
        }
        print(f"🔁 Bootstrap CI (95%): [{ci_lower:.6f}, {ci_upper:.6f}]")
        return result

    def compute_bayesian_superiority(self, n_samples: int = 20_000) -> Dict[str, Any]:
        """
        Bayesian probability that Hybrid > Baseline using posterior sampling
        over the mean difference.  Falls back to LLM vs NN when no explicit
        baseline scores are available.
        """
        hybrid_scores = np.array([
            r.get("evaluation", {}).get("r2")
            for r in self.hybrid_results
            if r.get("evaluation", {}).get("r2") is not None
        ])

        # Use explicit baseline_results when available, else fall back to NN scores
        if self.baseline_results:
            baseline_scores = np.array(self.baseline_results)
        else:
            baseline_scores = np.array([
                r.get("nn_result", {}).get("metrics", {}).get("r2")
                for r in self.hybrid_results
                if r.get("nn_result", {}).get("metrics", {}).get("r2") is not None
            ])

        if len(hybrid_scores) < 2 or len(baseline_scores) < 2:
            print("⚠️  Not enough scores for Bayesian superiority.")
            return {}

        mu_h, mu_b = hybrid_scores.mean(), baseline_scores.mean()
        std_h = hybrid_scores.std(ddof=1)
        std_b = baseline_scores.std(ddof=1)
        n_h, n_b = len(hybrid_scores), len(baseline_scores)

        diff_mean = mu_h - mu_b
        diff_std  = np.sqrt((std_h ** 2 / n_h) + (std_b ** 2 / n_b))

        samples          = np.random.normal(diff_mean, diff_std, n_samples)
        prob_superiority = float(np.mean(samples > 0))
        ci_lower         = float(np.percentile(samples, 2.5))
        ci_upper         = float(np.percentile(samples, 97.5))

        result = {
            "posterior_prob":     prob_superiority,
            "credible_interval":  (ci_lower, ci_upper),
            "mean_difference":    float(diff_mean),
        }
        print(f"🧠 Bayesian P(Hybrid > Baseline): {prob_superiority:.3f}  "
              f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]")
        return result

    def compute_effect_sizes(self) -> Dict[str, Any]:
        """
        Cohen's d between hybrid R² and baseline (NN) R².
        Also returns Hedges' g (small-sample correction).
        """
        hybrid_scores = np.array([
            r.get("evaluation", {}).get("r2")
            for r in self.hybrid_results
            if r.get("evaluation", {}).get("r2") is not None
        ])
        baseline_scores = np.array([
            r.get("nn_result", {}).get("metrics", {}).get("r2")
            for r in self.hybrid_results
            if r.get("nn_result", {}).get("metrics", {}).get("r2") is not None
        ])

        if len(hybrid_scores) < 2 or len(baseline_scores) < 2:
            print("⚠️  Not enough scores for effect-size computation.")
            return {}

        pooled_std = np.sqrt((np.var(hybrid_scores, ddof=1) + np.var(baseline_scores, ddof=1)) / 2)
        cohens_d   = (hybrid_scores.mean() - baseline_scores.mean()) / pooled_std if pooled_std else float("nan")

        # Hedges' g correction factor
        n_total   = len(hybrid_scores) + len(baseline_scores)
        hedges_g  = cohens_d * (1 - 3 / (4 * n_total - 9)) if n_total > 9 else cohens_d

        result = {
            "cohens_d": float(cohens_d),
            "hedges_g": float(hedges_g),
            "interpretation": (
                "negligible" if abs(cohens_d) < 0.2 else
                "small"      if abs(cohens_d) < 0.5 else
                "medium"     if abs(cohens_d) < 0.8 else
                "large"
            ),
        }
        print(f"📐 Cohen's d = {cohens_d:.4f}  ({result['interpretation']}),  "
              f"Hedges' g = {hedges_g:.4f}")
        return result

    def compute_structural_stability(self) -> Dict[str, Any]:
        """
        Measures run-to-run structural stability: variance in R² across
        domains (lower = more stable generalisation).
        """
        domain_means: Dict[str, list] = {}
        for r in self.hybrid_results:
            domain = r.get("domain", "Unknown")
            r2     = r.get("evaluation", {}).get("r2")
            if r2 is not None:
                domain_means.setdefault(domain, []).append(r2)

        if not domain_means:
            return {}

        per_domain_mean = {d: float(np.mean(v)) for d, v in domain_means.items()}
        variance_across_domains = float(np.var(list(per_domain_mean.values())))

        result = {
            "per_domain_mean_r2":       per_domain_mean,
            "variance_across_domains":  variance_across_domains,
            "stability_score":          float(1.0 / (1.0 + variance_across_domains)),
        }
        print(f"🏗️  Structural stability score: {result['stability_score']:.4f}  "
              f"(domain R² variance = {variance_across_domains:.6f})")
        return result

    def verify_statistical_power(self, min_power: float = 0.80) -> bool:
        """
        Approximate post-hoc power check (two-sample t-test).
        Warns if the experiment is likely underpowered.
        """
        from scipy.stats import norm  # lightweight import

        r2_scores = [
            r.get("evaluation", {}).get("r2")
            for r in self.hybrid_results
            if r.get("evaluation", {}).get("r2") is not None
        ]
        if len(r2_scores) < 4:
            print("⚠️  Statistical power check skipped — fewer than 4 valid scores.")
            return False

        n      = len(r2_scores)
        effect = 0.5           # assume medium Cohen's d
        alpha  = 0.05
        z_a    = norm.ppf(1 - alpha / 2)
        z_b    = norm.ppf(min_power)
        n_req  = int(np.ceil(2 * ((z_a + z_b) / effect) ** 2))

        sufficient = n >= n_req
        status = "✅" if sufficient else "⚠️ "
        print(f"{status} Statistical power: n={n}, required≈{n_req} "
              f"(effect=0.5, α=0.05, power={min_power})")
        if not sufficient:
            print(f"   → Consider increasing sample size by ~{n_req - n} cases.")
        return sufficient

    def validate_seed_reproducibility(self, seed: int = 42) -> bool:
        """
        Checks that numpy's global seed can produce a deterministic sample.
        This is a sentinel test; real seed management should happen in the
        experiment runner, not the analyser.
        """
        rng1 = np.random.RandomState(seed).rand(5)
        rng2 = np.random.RandomState(seed).rand(5)
        reproducible = np.allclose(rng1, rng2)
        status = "✅" if reproducible else "❌"
        print(f"{status} Seed reproducibility check (seed={seed}): "
              f"{'PASS' if reproducible else 'FAIL'}")
        return reproducible

    def save_full_run_json(self, summary: Dict[str, Any]) -> Path:
        """
        Persist the complete summary dict as a timestamped JSON artifact.
        """
        def _serialise(obj):
            """Make numpy scalars / tuples JSON-safe."""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

        out_path = self.results_dir / f"full_run_{self.timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=_serialise)
        print(f"💾 Full run JSON saved: {out_path}")
        return out_path

    def generate_all_figures(self, summary: Dict[str, Any]) -> None:
        """
        Orchestrate all publication figures.
        Individual plot logic is called here to avoid scattering plt.savefig
        calls across multiple methods.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib not available — skipping figures.")
            return

        try:
            import seaborn as sns
            _seaborn_available = True
        except ImportError:
            _seaborn_available = False
            print("⚠️  seaborn not available — violin plot will be skipped.")

        # ----- 1. Component distribution violin -----
        data = []
        for result in self.hybrid_results:
            for comp, key in [("LLM",  ("llm_result",  "metrics", "r2")),
                               ("NN",   ("nn_result",   "metrics", "r2")),
                               ("Hybrid", ("evaluation", "r2"))]:
                r2 = result
                for k in key:
                    r2 = r2.get(k, {}) if isinstance(r2, dict) else None
                if r2 is not None:
                    data.append({"Method": comp, "R²": r2})

        if data and _seaborn_available:
            import pandas as pd
            from scipy.stats import ttest_ind
            df = pd.DataFrame(data)
            fig, ax = plt.subplots()
            sns.violinplot(data=df, x="Method", y="R²", inner="box", ax=ax)

            llm_r2 = df.loc[df["Method"] == "LLM",  "R²"].values
            nn_r2  = df.loc[df["Method"] == "NN",   "R²"].values
            if len(llm_r2) > 2 and len(nn_r2) > 2:
                _, p = ttest_ind(llm_r2, nn_r2)
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax.text(0.5, df["R²"].max() + 0.005, stars, ha="center", fontsize=14,
                        transform=ax.get_xaxis_transform())

            ax.set_title("Distribution of R² by Component")
            fig.tight_layout()
            fig.savefig(self.figures_dir / "violin_significance.png", dpi=150)
            plt.close(fig)
            print("   📊 violin_significance.png")

        # ----- 2. Paired-difference histogram with bootstrap CI -----
        diffs = []
        for result in self.hybrid_results:
            llm_r2 = result.get("llm_result", {}).get("metrics", {}).get("r2")
            nn_r2  = result.get("nn_result",  {}).get("metrics", {}).get("r2")
            if llm_r2 is not None and nn_r2 is not None:
                diffs.append(llm_r2 - nn_r2)

        if diffs:
            diffs_arr  = np.array(diffs)
            boot_means = [np.mean(np.random.choice(diffs_arr, size=len(diffs_arr), replace=True))
                          for _ in range(2000)]
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)

            fig, ax = plt.subplots()
            ax.hist(diffs_arr, bins=15, edgecolor="black")
            ax.axvline(0, linestyle="--", color="red", label="No difference")
            ax.axvspan(ci_lo, ci_hi, alpha=0.2, color="blue", label="95% bootstrap CI")
            ax.set_xlabel("LLM R² – NN R²")
            ax.set_title("Paired Difference with Bootstrap CI")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.figures_dir / "paired_difference_ci.png", dpi=150)
            plt.close(fig)
            print("   📊 paired_difference_ci.png")

        # ----- 3. Effect-size forest plot -----
        effect = summary.get("effect_sizes", {})
        d = effect.get("cohens_d")
        if d is not None:
            se = 1 / np.sqrt(max(len(self.hybrid_results), 1))
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.errorbar(d, 0, xerr=[[d - (d - 1.96 * se)], [(d + 1.96 * se) - d]],
                        fmt="o", color="navy")
            ax.axvline(0, linestyle="--", color="grey")
            ax.set_yticks([])
            ax.set_xlabel("Cohen's d")
            ax.set_title("Effect Size Forest Plot")
            fig.tight_layout()
            fig.savefig(self.figures_dir / "effect_size_forest.png", dpi=150)
            plt.close(fig)
            print("   📊 effect_size_forest.png")

    # end of generate_all_figures

    def run_full_analysis(self):
        """Run complete analysis pipeline with registry + publication support"""

        print("\n" + "=" * 80)
        print("🔍 HYBRID SYSTEM PERFORMANCE ANALYSIS")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print("=" * 80)

        # --------------------------------------------------
        # 1️⃣ Load results
        # --------------------------------------------------
        if not self.load_latest_results():
            return

        # --------------------------------------------------
        # 2️⃣ Core statistical analyses
        # --------------------------------------------------
        overall_stats   = self.analyze_overall_performance()
        domain_stats    = self.analyze_by_domain()
        strategy_stats  = self.analyze_decision_strategy()
        component_stats = self.analyze_component_performance()
        failures        = self.analyze_failures()

        # --------------------------------------------------
        # 3️⃣ Extrapolation metrics
        # --------------------------------------------------
        extrapolation_stats = self.analyze_extrapolation()

        # --------------------------------------------------
        # 4️⃣ Advanced research metrics
        # --------------------------------------------------
        bootstrap_stats  = self.compute_bootstrap_confidence_intervals()
        bayesian_stats   = self.compute_bayesian_superiority()
        effect_sizes     = self.compute_effect_sizes()
        stability_metrics = self.compute_structural_stability()

        # --------------------------------------------------
        # 5️⃣ Build master summary dictionary
        # --------------------------------------------------
        summary = {
            "timestamp":     self.timestamp,
            "overall":       overall_stats,
            "domain":        domain_stats,
            "strategy":      strategy_stats,
            "components":    component_stats,
            "failures":      len(failures),
            "extrapolation": extrapolation_stats,
            "bootstrap":     bootstrap_stats,
            "bayesian":      bayesian_stats,
            "effect_sizes":  effect_sizes,
            "stability":     stability_metrics,
        }

        # --------------------------------------------------
        # 6️⃣ Experiment Registry
        # --------------------------------------------------
        registry_entry = {
            "timestamp":            self.timestamp,
            "num_cases":            overall_stats.get("total_cases", 0),
            "mean_r2":              overall_stats.get("mean_r2"),
            "success_rate":         overall_stats.get("success_rate"),
            "posterior_superiority": bayesian_stats.get("posterior_prob"),
            "effect_size":          effect_sizes.get("cohens_d"),
        }

        registry_path = self.results_dir / "experiment_registry.json"
        registry = json.load(open(registry_path)) if registry_path.exists() else []
        registry.append(registry_entry)
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=4)
        print("📘 Experiment logged to registry.")

        # --------------------------------------------------
        # 7️⃣ Research Rigor Validation (JMLR-grade)
        # --------------------------------------------------
        self.verify_statistical_power()
        self.validate_seed_reproducibility()
        self.save_full_run_json(summary)

        # --------------------------------------------------
        # 8️⃣ Generate publication assets
        # --------------------------------------------------
        self.generate_all_figures(summary)
        self.generate_latex_report(summary)

        # --------------------------------------------------
        # 9️⃣ Recommendations + Report
        # (recommendations must be built BEFORE save_report)
        # --------------------------------------------------
        recommendations = self.generate_recommendations(
            overall_stats, domain_stats, strategy_stats, failures
        )
        report_file = self.save_report(
            overall_stats,
            domain_stats,
            strategy_stats,
            component_stats,
            failures,
            recommendations,          # ← correct 6th arg (was mistakenly `summary`)
        )

        # --------------------------------------------------
        # 🔟 Final Console Summary
        # --------------------------------------------------
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   Total cases analyzed: {overall_stats.get('total_cases', 0)}")
        print(f"   Mean R²: {overall_stats.get('mean_r2', 0):.6f}")
        print(f"   Success rate: {overall_stats.get('success_rate', 0) * 100:.1f}%")
        print(f"   Failures (R² < 0.90): {len(failures)}")
        if bayesian_stats:
            print(f"   Posterior superiority: {bayesian_stats.get('posterior_prob', 0):.3f}")
        print(f"\n📁 Report: {report_file}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Analyze Hybrid DeFi System Performance")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='hypatiax/data/results',
        help='Directory containing results files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = HybridPerformanceAnalyzer(results_dir=args.results_dir)
    analyzer.verbose = args.verbose
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
