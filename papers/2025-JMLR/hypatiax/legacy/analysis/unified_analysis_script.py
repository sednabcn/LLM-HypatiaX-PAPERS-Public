#!/usr/bin/env python3
"""
Unified Analysis Script - Data Merger and Preprocessor
======================================================
Merges multiple JSON data files into a single unified format
for downstream visualization and analysis.

This script:
1. Loads data from multiple sources (extrapolation, interpolation, systems)
2. Validates and cleans the data
3. Merges into a unified format
4. Saves as all_systems_merged.json
5. Provides summary statistics

Usage:
    python unified_analysis_script.py
    python unified_analysis_script.py --input-dir /path/to/data
    python unified_analysis_script.py --output merged_data.json
    python unified_analysis_script.py --verbose

Author: Generated for HypatiaX DeFi Analysis
Date: February 2026
Version: 1.0
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import argparse
from collections import defaultdict


class UnifiedDataMerger:
    """Merges multiple data sources into unified format"""

    def __init__(self, input_dir: Path = None, verbose: bool = False):
        """
        Initialize the data merger
        
        Args:
            input_dir: Directory containing input JSON files (default: current dir)
            verbose: Print detailed progress information
        """
        self.input_dir = input_dir or Path.cwd()
        self.verbose = verbose
        
        # Data storage
        self.extrapolation_data = None
        self.interpolation_data = None
        self.systems_data = None
        
        # Merged result
        self.merged_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "source_files": [],
                "merger_version": "1.0",
            },
            "tests": [],
            "total_tests": 0,
            "systems": [],
        }
        
        # Statistics tracking
        self.stats = {
            "files_loaded": 0,
            "tests_processed": 0,
            "methods_found": set(),
            "domains_found": set(),
        }

    def log(self, message: str, level: str = "INFO"):
        """Print log message if verbose"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            prefix = {
                "INFO": "ℹ️ ",
                "SUCCESS": "✅",
                "WARNING": "⚠️ ",
                "ERROR": "❌",
            }.get(level, "  ")
            print(f"{prefix} {message}")

    def find_data_files(self) -> Dict[str, Optional[Path]]:
        """
        Find available data files in input directory
        
        Returns:
            Dictionary mapping data type to file path
        """
        self.log("Searching for data files...", "INFO")
        
        # File patterns to search for (in priority order)
        file_patterns = {
            "extrapolation": [
                "all_domains_extrap_v4_*.json",
                "extrapolation_*.json",
                "*extrap*.json",
            ],
            "interpolation": [
                "results_*.json",
                "standalone_real_methods_*.json",
                "interpolation_*.json",
                "*interp*.json",
            ],
            "systems": [
                "systems_2_3_2_data.json",
                "systems_2_3_data.json",
                "systems_*.json",
            ],
        }
        
        found_files = {}
        
        for data_type, patterns in file_patterns.items():
            found = None
            for pattern in patterns:
                matches = list(self.input_dir.glob(pattern))
                if matches:
                    # Use most recent file if multiple matches
                    found = max(matches, key=lambda p: p.stat().st_mtime)
                    break
            
            found_files[data_type] = found
            
            if found:
                self.log(f"Found {data_type}: {found.name}", "SUCCESS")
                self.merged_data["metadata"]["source_files"].append(str(found.name))
            else:
                self.log(f"No {data_type} file found", "WARNING")
        
        return found_files

    def load_json_file(self, filepath: Path) -> Optional[Dict]:
        """
        Safely load JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded data or None if failed
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.log(f"Loaded {filepath.name} ({filepath.stat().st_size / 1024:.1f} KB)", "SUCCESS")
            self.stats["files_loaded"] += 1
            return data
        except json.JSONDecodeError as e:
            self.log(f"JSON decode error in {filepath.name}: {e}", "ERROR")
            return None
        except Exception as e:
            self.log(f"Error loading {filepath.name}: {e}", "ERROR")
            return None

    def normalize_method_name(self, method_name: str) -> str:
        """
        Normalize method names to standard format
        
        Args:
            method_name: Original method name
            
        Returns:
            Normalized method name
        """
        # Mapping of various naming conventions to standard names
        name_mappings = {
            # Hybrid variations
            "hybrid system v40": "Hybrid System v40",
            "hybrid_v40": "Hybrid System v40",
            "hybrid v40": "Hybrid System v40",
            "hybrid": "Hybrid System v40",
            
            # LLM variations
            "pure llm": "Pure LLM (Enhanced)",
            "pure_llm": "Pure LLM (Enhanced)",
            "llm": "Pure LLM (Enhanced)",
            "pure llm (enhanced)": "Pure LLM (Enhanced)",
            "pure llm (basic)": "Pure LLM (Basic)",
            
            # Neural Network variations
            "neural network": "Neural Network",
            "neural_network": "Neural Network",
            "nn": "Neural Network",
            
            # Ensemble variations
            "ensemble": "LLM+NN Ensemble (Smart)",
            "ensemble smart": "LLM+NN Ensemble (Smart)",
            "llm+nn ensemble": "LLM+NN Ensemble (Smart)",
            "llm+nn ensemble (smart)": "LLM+NN Ensemble (Smart)",
            
            # System variations
            "system 3": "System 3 LLM+Fallback",
            "system_3": "System 3 LLM+Fallback",
            "system 3 llm+fallback": "System 3 LLM+Fallback",
            "system 2": "System 2 Symbolic",
            "system_2": "System 2 Symbolic",
            "system 2 symbolic": "System 2 Symbolic",
        }
        
        normalized = name_mappings.get(method_name.lower(), method_name)
        self.stats["methods_found"].add(normalized)
        return normalized

    def extract_test_from_extrapolation(self, test_data: Dict, test_name: str = None) -> Dict:
        """
        Extract test information from extrapolation data format
        
        Args:
            test_data: Raw test data
            test_name: Name of the test
            
        Returns:
            Standardized test dictionary
        """
        test = {
            "test_name": test_name or test_data.get("test_name", "unknown"),
            "domain": test_data.get("domain", "unknown"),
            "description": test_data.get("description", ""),
            "results": {},
        }
        
        self.stats["domains_found"].add(test["domain"])
        
        # Extract results for each method
        results_data = test_data.get("results", {})
        
        for method_name, method_result in results_data.items():
            normalized_name = self.normalize_method_name(method_name)
            
            # Extract metrics
            result_entry = {
                "r2": method_result.get("r2", method_result.get("r2_score", np.nan)),
                "rmse": method_result.get("rmse", np.nan),
                "success": method_result.get("success", False),
            }
            
            # Extract extrapolation errors if available
            if "extrapolation_errors" in method_result:
                result_entry["extrapolation_errors"] = method_result["extrapolation_errors"]
            elif "medium" in method_result or "medium_2x" in method_result:
                result_entry["extrapolation_errors"] = {
                    "easy": method_result.get("easy", method_result.get("easy_1.5x", np.nan)),
                    "medium": method_result.get("medium", method_result.get("medium_2x", np.nan)),
                    "hard": method_result.get("hard", method_result.get("hard_3x", np.nan)),
                }
            
            # Extract formula if available
            if "formula" in method_result:
                result_entry["formula"] = method_result["formula"]
            
            test["results"][normalized_name] = result_entry
        
        return test

    def extract_test_from_interpolation(self, test_data: Dict, test_name: str = None) -> Dict:
        """
        Extract test information from interpolation data format
        
        Args:
            test_data: Raw test data
            test_name: Name of the test
            
        Returns:
            Standardized test dictionary
        """
        test = {
            "test_name": test_name or test_data.get("name", "unknown"),
            "domain": test_data.get("domain", "unknown"),
            "description": test_data.get("description", ""),
            "results": {},
        }
        
        self.stats["domains_found"].add(test["domain"])
        
        # Handle different interpolation result structures
        if "methods" in test_data:
            methods_data = test_data["methods"]
        elif "results" in test_data:
            methods_data = test_data["results"]
        else:
            methods_data = {k: v for k, v in test_data.items() if isinstance(v, dict)}
        
        for method_name, method_result in methods_data.items():
            if not isinstance(method_result, dict):
                continue
                
            normalized_name = self.normalize_method_name(method_name)
            
            result_entry = {
                "r2": method_result.get("r2", method_result.get("r2_score", method_result.get("train_r2", np.nan))),
                "rmse": method_result.get("rmse", method_result.get("train_rmse", np.nan)),
                "success": method_result.get("success", True),
            }
            
            # Add validation metrics if available
            if "validation" in method_result:
                result_entry["validation_r2"] = method_result["validation"].get("r2", np.nan)
            
            test["results"][normalized_name] = result_entry
        
        return test

    def merge_test_results(self, test1: Dict, test2: Dict) -> Dict:
        """
        Merge two test dictionaries (same test from different sources)
        
        Args:
            test1: First test dictionary
            test2: Second test dictionary
            
        Returns:
            Merged test dictionary
        """
        # Use test1 as base
        merged = test1.copy()
        
        # Merge results, preferring non-NaN values
        for method_name, result2 in test2.get("results", {}).items():
            if method_name in merged["results"]:
                # Merge existing method
                result1 = merged["results"][method_name]
                for key, value2 in result2.items():
                    if key not in result1 or (isinstance(result1[key], float) and np.isnan(result1[key])):
                        result1[key] = value2
                    elif isinstance(value2, dict) and isinstance(result1[key], dict):
                        # Merge nested dicts (e.g., extrapolation_errors)
                        result1[key].update(value2)
            else:
                # Add new method
                merged["results"][method_name] = result2
        
        # Merge other fields
        if test2.get("description") and not merged.get("description"):
            merged["description"] = test2["description"]
        
        return merged

    def process_extrapolation_data(self):
        """Process extrapolation test data"""
        if not self.extrapolation_data:
            return
        
        self.log("Processing extrapolation data...", "INFO")
        
        # Handle different extrapolation data structures
        if "tests" in self.extrapolation_data:
            tests = self.extrapolation_data["tests"]
        elif isinstance(self.extrapolation_data, list):
            tests = self.extrapolation_data
        elif "results" in self.extrapolation_data:
            # Convert results dict to list of tests
            tests = []
            for test_name, test_data in self.extrapolation_data["results"].items():
                test_data["test_name"] = test_name
                tests.append(test_data)
        else:
            self.log("Unknown extrapolation data structure", "WARNING")
            return
        
        for test_data in tests:
            if isinstance(test_data, dict):
                test = self.extract_test_from_extrapolation(test_data)
                self.merged_data["tests"].append(test)
                self.stats["tests_processed"] += 1

    def process_interpolation_data(self):
        """Process interpolation test data"""
        if not self.interpolation_data:
            return
        
        self.log("Processing interpolation data...", "INFO")
        
        # Handle different interpolation data structures
        if "tests" in self.interpolation_data:
            tests = self.interpolation_data["tests"]
        elif "results" in self.interpolation_data:
            tests_dict = self.interpolation_data["results"]
            tests = []
            for test_name, test_data in tests_dict.items():
                if isinstance(test_data, dict):
                    test_data["test_name"] = test_name
                    tests.append(test_data)
        elif isinstance(self.interpolation_data, list):
            tests = self.interpolation_data
        else:
            self.log("Unknown interpolation data structure", "WARNING")
            return
        
        for test_data in tests:
            if not isinstance(test_data, dict):
                continue
            
            test = self.extract_test_from_interpolation(test_data)
            
            # Try to merge with existing test
            test_name = test["test_name"]
            existing_idx = None
            
            for idx, existing_test in enumerate(self.merged_data["tests"]):
                if existing_test["test_name"] == test_name:
                    existing_idx = idx
                    break
            
            if existing_idx is not None:
                # Merge with existing
                self.merged_data["tests"][existing_idx] = self.merge_test_results(
                    self.merged_data["tests"][existing_idx], test
                )
            else:
                # Add new test
                self.merged_data["tests"].append(test)
                self.stats["tests_processed"] += 1

    def process_systems_data(self):
        """Process system comparison data"""
        if not self.systems_data:
            return
        
        self.log("Processing systems data...", "INFO")
        
        # Add systems metadata
        if "systems" in self.systems_data:
            self.merged_data["systems"] = self.systems_data["systems"]
        
        # Process any additional test data
        if "tests" in self.systems_data:
            for test_data in self.systems_data["tests"]:
                if isinstance(test_data, dict):
                    test = self.extract_test_from_extrapolation(test_data)
                    
                    # Try to merge with existing
                    test_name = test["test_name"]
                    merged = False
                    
                    for idx, existing_test in enumerate(self.merged_data["tests"]):
                        if existing_test["test_name"] == test_name:
                            self.merged_data["tests"][idx] = self.merge_test_results(
                                existing_test, test
                            )
                            merged = True
                            break
                    
                    if not merged:
                        self.merged_data["tests"].append(test)
                        self.stats["tests_processed"] += 1

    def validate_merged_data(self) -> Tuple[bool, List[str]]:
        """
        Validate the merged data
        
        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []
        
        # Check for tests
        if not self.merged_data["tests"]:
            warnings.append("No tests found in merged data")
            return False, warnings
        
        # Check for methods
        methods_found = set()
        for test in self.merged_data["tests"]:
            methods_found.update(test.get("results", {}).keys())
        
        if not methods_found:
            warnings.append("No method results found")
            return False, warnings
        
        # Check for data completeness
        tests_with_r2 = sum(
            1 for test in self.merged_data["tests"]
            if any(
                isinstance(r.get("r2"), (int, float)) and not np.isnan(r.get("r2"))
                for r in test.get("results", {}).values()
            )
        )
        
        if tests_with_r2 == 0:
            warnings.append("No tests with valid R² scores found")
        
        tests_with_extrap = sum(
            1 for test in self.merged_data["tests"]
            if any(
                "extrapolation_errors" in r
                for r in test.get("results", {}).values()
            )
        )
        
        if tests_with_extrap == 0:
            warnings.append("No tests with extrapolation data found")
        
        return True, warnings

    def generate_summary(self) -> Dict:
        """
        Generate summary statistics
        
        Returns:
            Dictionary of summary statistics
        """
        summary = {
            "total_tests": len(self.merged_data["tests"]),
            "total_methods": len(self.stats["methods_found"]),
            "methods": sorted(list(self.stats["methods_found"])),
            "domains": sorted(list(self.stats["domains_found"])),
            "source_files": self.merged_data["metadata"]["source_files"],
        }
        
        # Calculate R² statistics per method
        method_stats = defaultdict(lambda: {"r2_values": [], "extrap_values": []})
        
        for test in self.merged_data["tests"]:
            for method_name, result in test.get("results", {}).items():
                r2 = result.get("r2")
                if isinstance(r2, (int, float)) and not np.isnan(r2):
                    method_stats[method_name]["r2_values"].append(r2)
                
                if "extrapolation_errors" in result:
                    medium_error = result["extrapolation_errors"].get("medium")
                    if isinstance(medium_error, (int, float)) and not np.isnan(medium_error):
                        method_stats[method_name]["extrap_values"].append(medium_error)
        
        # Compute statistics
        summary["method_statistics"] = {}
        for method_name, stats in method_stats.items():
            summary["method_statistics"][method_name] = {
                "r2_count": len(stats["r2_values"]),
                "r2_mean": float(np.mean(stats["r2_values"])) if stats["r2_values"] else None,
                "r2_median": float(np.median(stats["r2_values"])) if stats["r2_values"] else None,
                "extrap_count": len(stats["extrap_values"]),
                "extrap_median": float(np.median(stats["extrap_values"])) if stats["extrap_values"] else None,
            }
        
        return summary

    def save_merged_data(self, output_path: Path):
        """
        Save merged data to JSON file
        
        Args:
            output_path: Path to save merged data
        """
        # Update metadata
        self.merged_data["total_tests"] = len(self.merged_data["tests"])
        self.merged_data["summary"] = self.generate_summary()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.merged_data, f, indent=2)
            
            file_size = output_path.stat().st_size / 1024
            self.log(f"Saved merged data to {output_path.name} ({file_size:.1f} KB)", "SUCCESS")
        except Exception as e:
            self.log(f"Error saving merged data: {e}", "ERROR")
            raise

    def print_summary(self):
        """Print summary of merged data"""
        print("\n" + "=" * 80)
        print("MERGE SUMMARY")
        print("=" * 80)
        
        summary = self.merged_data.get("summary", self.generate_summary())
        
        print(f"\n📊 Data Sources:")
        for source in summary["source_files"]:
            print(f"   • {source}")
        
        print(f"\n📈 Statistics:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Total methods: {summary['total_methods']}")
        print(f"   Domains: {len(summary['domains'])}")
        
        print(f"\n🔧 Methods found:")
        for method in summary["methods"]:
            print(f"   • {method}")
        
        print(f"\n🏷️  Domains:")
        for domain in summary["domains"]:
            print(f"   • {domain}")
        
        print(f"\n📊 Method Statistics:")
        for method, stats in summary.get("method_statistics", {}).items():
            print(f"\n   {method}:")
            if stats["r2_count"] > 0:
                print(f"      R²: {stats['r2_count']} tests, mean={stats['r2_mean']:.4f}, median={stats['r2_median']:.4f}")
            if stats["extrap_count"] > 0:
                print(f"      Extrapolation: {stats['extrap_count']} tests, median error={stats['extrap_median']:.1f}%")

    def run_merge(self, output_path: Path = None) -> bool:
        """
        Run the complete merge pipeline
        
        Args:
            output_path: Path to save merged data (default: all_systems_merged.json)
            
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("UNIFIED DATA MERGER")
        print("=" * 80)
        print(f"\nInput directory: {self.input_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")
        
        # Find files
        files = self.find_data_files()
        
        if not any(files.values()):
            self.log("No data files found!", "ERROR")
            return False
        
        # Load data
        if files["extrapolation"]:
            self.extrapolation_data = self.load_json_file(files["extrapolation"])
        
        if files["interpolation"]:
            self.interpolation_data = self.load_json_file(files["interpolation"])
        
        if files["systems"]:
            self.systems_data = self.load_json_file(files["systems"])
        
        # Process data
        self.process_extrapolation_data()
        self.process_interpolation_data()
        self.process_systems_data()
        
        # Validate
        is_valid, warnings = self.validate_merged_data()
        
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"   • {warning}")
        
        if not is_valid:
            self.log("Validation failed - merged data may be incomplete", "ERROR")
            return False
        
        # Save
        output_path = output_path or (self.input_dir / "all_systems_merged.json")
        self.save_merged_data(output_path)
        
        # Print summary
        self.print_summary()
        
        print("\n" + "=" * 80)
        print("✅ MERGE COMPLETE")
        print("=" * 80)
        print(f"\n📁 Output: {output_path}")
        print(f"📊 Tests: {len(self.merged_data['tests'])}")
        print(f"🔧 Methods: {len(self.stats['methods_found'])}")
        print("\n💡 Next steps:")
        print("   python create_visualizations.py")
        print("=" * 80 + "\n")
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Merge multiple data sources into unified format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_analysis_script.py
  python unified_analysis_script.py --input-dir ./results
  python unified_analysis_script.py --output custom_merged.json
  python unified_analysis_script.py --verbose
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing input JSON files (default: current directory)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: all_systems_merged.json)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir) if args.input_dir else Path.cwd()
    output_path = Path(args.output) if args.output else None
    
    # Run merger
    merger = UnifiedDataMerger(input_dir=input_dir, verbose=args.verbose)
    
    try:
        success = merger.run_merge(output_path=output_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
