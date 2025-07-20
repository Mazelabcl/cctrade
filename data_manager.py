# data_manager.py

import json
import os
import pandas as pd
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional, Tuple
import logging

class DataManager:
    """Manages time-chunked datasets with manifest tracking and validation."""
    
    def __init__(self, manifest_path: str = "data_manifest.json", base_data_dir: str = "datasets"):
        self.manifest_path = manifest_path
        self.base_data_dir = base_data_dir
        self.manifest = self._load_or_create_manifest()
    
    def _load_or_create_manifest(self) -> Dict:
        """Load existing manifest or create a new one by scanning data directory."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    manifest = json.load(f)
                    print(f"Loaded existing manifest with {len(manifest.get('datasets', []))} periods")
                    return manifest
            except Exception as e:
                print(f"Error loading manifest: {e}. Creating new one.")
        
        # Create new manifest by scanning directory
        manifest = self._scan_and_create_manifest()
        self._save_manifest(manifest)
        return manifest
    
    def _scan_and_create_manifest(self) -> Dict:
        """Scan base_data directory and create manifest from existing files."""
        datasets = []
        
        if not os.path.exists(self.base_data_dir):
            print(f"Base data directory {self.base_data_dir} not found")
            return {"datasets": [], "default_period": None, "last_updated": datetime.now().isoformat()}
        
        # Pattern to match time-chunked files: dataset_YYYY_MM_DD-YYYY_MM_DD.csv
        pattern = r'(\w+)_(\d{4}_\d{2}_\d{2})-(\d{4}_\d{2}_\d{2})\.csv'
        
        files = os.listdir(self.base_data_dir)
        periods = {}
        
        for file in files:
            match = re.match(pattern, file)
            if match:
                dataset_type, start_date_str, end_date_str = match.groups()
                
                # Convert date format from YYYY_MM_DD to YYYY-MM-DD
                start_date = start_date_str.replace('_', '-')
                end_date = end_date_str.replace('_', '-')
                period_key = f"{start_date_str}-{end_date_str}"
                
                if period_key not in periods:
                    periods[period_key] = {
                        "period": period_key,
                        "start_date": start_date,
                        "end_date": end_date,
                        "files": {}
                    }
                
                periods[period_key]["files"][dataset_type] = {
                    "path": os.path.join(self.base_data_dir, file),
                    "size": os.path.getsize(os.path.join(self.base_data_dir, file))
                }
        
        # Convert to datasets format and get row counts
        for period_data in periods.values():
            try:
                # Get row count from ml_dataset if available
                rows = 0
                if "ml_dataset" in period_data["files"]:
                    ml_path = period_data["files"]["ml_dataset"]["path"]
                    rows = sum(1 for _ in open(ml_path)) - 1  # Subtract header
                
                dataset = {
                    "period": period_data["period"],
                    "start_date": period_data["start_date"],
                    "end_date": period_data["end_date"],
                    "files": period_data["files"],
                    "rows": rows,
                    "last_updated": datetime.fromtimestamp(
                        max(os.path.getmtime(f["path"]) for f in period_data["files"].values())
                    ).isoformat(),
                    "status": "complete" if len(period_data["files"]) >= 2 else "incomplete"
                }
                datasets.append(dataset)
                
            except Exception as e:
                print(f"Error processing period {period_data['period']}: {e}")
        
        # Sort by start date and set default to most recent
        datasets.sort(key=lambda x: x["start_date"])
        default_period = datasets[-1]["period"] if datasets else None
        
        manifest = {
            "datasets": datasets,
            "default_period": default_period,
            "last_updated": datetime.now().isoformat(),
            "base_data_dir": self.base_data_dir
        }
        
        print(f"Created manifest with {len(datasets)} periods")
        return manifest
    
    def _save_manifest(self, manifest: Dict = None):
        """Save manifest to file."""
        if manifest is None:
            manifest = self.manifest
        
        manifest["last_updated"] = datetime.now().isoformat()
        
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def get_available_periods(self) -> List[str]:
        """Get list of available data periods."""
        return [dataset["period"] for dataset in self.manifest["datasets"]]
    
    def get_period_info(self, period: str = None) -> Optional[Dict]:
        """Get information about a specific period or default."""
        if period is None:
            period = self.manifest.get("default_period")
        
        for dataset in self.manifest["datasets"]:
            if dataset["period"] == period:
                return dataset
        return None
    
    def validate_data_integrity(self) -> Dict:
        """Validate data integrity, check for gaps and overlaps."""
        issues = {
            "missing_files": [],
            "gaps": [],
            "overlaps": [],
            "invalid_periods": []
        }
        
        datasets = sorted(self.manifest["datasets"], key=lambda x: x["start_date"])
        
        # Check file existence
        for dataset in datasets:
            for file_type, file_info in dataset["files"].items():
                if not os.path.exists(file_info["path"]):
                    issues["missing_files"].append({
                        "period": dataset["period"],
                        "file_type": file_type,
                        "path": file_info["path"]
                    })
        
        # Check for gaps and overlaps
        for i in range(len(datasets) - 1):
            current = datasets[i]
            next_dataset = datasets[i + 1]
            
            current_end = datetime.strptime(current["end_date"], "%Y-%m-%d")
            next_start = datetime.strptime(next_dataset["start_date"], "%Y-%m-%d")
            
            # Check for gaps (more than 1 day between periods)
            if (next_start - current_end).days > 1:
                issues["gaps"].append({
                    "after_period": current["period"],
                    "before_period": next_dataset["period"],
                    "gap_days": (next_start - current_end).days - 1
                })
            
            # Check for overlaps
            elif next_start <= current_end:
                issues["overlaps"].append({
                    "period1": current["period"],
                    "period2": next_dataset["period"],
                    "overlap_days": (current_end - next_start).days + 1
                })
        
        return issues
    
    def get_period_for_quick_test(self, months: int = 3) -> Optional[str]:
        """Get a small period suitable for quick testing."""
        # Try to find a recent period with limited data
        datasets = sorted(self.manifest["datasets"], key=lambda x: x["start_date"], reverse=True)
        
        for dataset in datasets:
            if dataset["rows"] > 0:
                # Calculate approximate months based on hourly data
                # ~730 hours per month, so target_rows = months * 730
                target_rows = months * 730
                if dataset["rows"] <= target_rows * 1.5:  # Allow 50% tolerance
                    return dataset["period"]
        
        # If no suitable period found, suggest creating one
        return None
    
    def create_sample_period(self, source_period: str, sample_rows: int, output_suffix: str = "sample") -> str:
        """Create a sample dataset from an existing period."""
        source_info = self.get_period_info(source_period)
        if not source_info:
            raise ValueError(f"Source period {source_period} not found")
        
        # Create sample files
        sample_period = f"{source_period}_{output_suffix}"
        
        for file_type, file_info in source_info["files"].items():
            if not os.path.exists(file_info["path"]):
                continue
                
            # Read and sample data
            df = pd.read_csv(file_info["path"])
            sample_df = df.head(sample_rows)
            
            # Create sample file
            sample_filename = f"{file_type}_{sample_period}.csv"
            sample_path = os.path.join(self.base_data_dir, sample_filename)
            sample_df.to_csv(sample_path, index=False)
            
            print(f"Created sample file: {sample_path} ({len(sample_df)} rows)")
        
        # Update manifest with sample period
        start_date = source_info["start_date"]
        sample_end_date = pd.read_csv(source_info["files"]["ml_dataset"]["path"]).iloc[sample_rows-1]["open_time"][:10]
        
        sample_dataset = {
            "period": sample_period,
            "start_date": start_date,
            "end_date": sample_end_date,
            "files": {
                file_type: {
                    "path": os.path.join(self.base_data_dir, f"{file_type}_{sample_period}.csv"),
                    "size": os.path.getsize(os.path.join(self.base_data_dir, f"{file_type}_{sample_period}.csv"))
                }
                for file_type in source_info["files"].keys()
            },
            "rows": sample_rows,
            "last_updated": datetime.now().isoformat(),
            "status": "sample",
            "source_period": source_period
        }
        
        self.manifest["datasets"].append(sample_dataset)
        self._save_manifest()
        
        return sample_period
    
    def print_data_status(self):
        """Print comprehensive data status report."""
        print("\n" + "="*60)
        print("DATA MANIFEST STATUS")
        print("="*60)
        
        # Available periods
        periods = self.get_available_periods()
        print(f"\nAvailable Periods ({len(periods)}):")
        for dataset in sorted(self.manifest["datasets"], key=lambda x: x["start_date"]):
            status_icon = "✓" if dataset["status"] == "complete" else "⚠"
            sample_note = " (SAMPLE)" if dataset["status"] == "sample" else ""
            print(f"  {status_icon} {dataset['period']}: {dataset['start_date']} to {dataset['end_date']} "
                  f"({dataset['rows']:,} rows){sample_note}")
        
        # Default period
        default = self.manifest.get("default_period")
        print(f"\nDefault Period: {default}")
        
        # Data validation
        print(f"\nData Validation:")
        issues = self.validate_data_integrity()
        
        if not any(issues.values()):
            print("  ✓ No issues found")
        else:
            if issues["missing_files"]:
                print(f"  ⚠ Missing files: {len(issues['missing_files'])}")
            if issues["gaps"]:
                print(f"  ⚠ Data gaps: {len(issues['gaps'])}")
                for gap in issues["gaps"]:
                    print(f"    - {gap['gap_days']} days between {gap['after_period']} and {gap['before_period']}")
            if issues["overlaps"]:
                print(f"  ⚠ Data overlaps: {len(issues['overlaps'])}")
                for overlap in issues["overlaps"]:
                    print(f"    - {overlap['overlap_days']} days overlap between {overlap['period1']} and {overlap['period2']}")
        
        # Quick test recommendation
        quick_period = self.get_period_for_quick_test()
        if quick_period:
            quick_info = self.get_period_info(quick_period)
            print(f"\nQuick Test Recommendation:")
            print(f"  Period: {quick_period} ({quick_info['rows']:,} rows)")
            print(f"  Usage: python main.py --period {quick_period} --features-only")
        
        # Data freshness analysis
        self._print_data_freshness_analysis()
        
        print("="*60)
    
    def _print_data_freshness_analysis(self):
        """Analyze data freshness and suggest updates."""
        from datetime import datetime, timedelta
        
        if not self.manifest["datasets"]:
            return
        
        print(f"\nData Freshness Analysis:")
        
        # Find most recent data end date
        latest_dataset = max(self.manifest["datasets"], key=lambda x: x["end_date"])
        latest_end = datetime.strptime(latest_dataset["end_date"], "%Y-%m-%d")
        current_date = datetime.now()
        
        gap_days = (current_date - latest_end).days
        
        if gap_days <= 7:
            print(f"  ✓ Data is current (last update: {latest_dataset['end_date']})")
        elif gap_days <= 30:
            print(f"  ⚠ Data is {gap_days} days old (last: {latest_dataset['end_date']})")
            print(f"    Consider updating soon")
        else:
            gap_months = gap_days / 30.44
            print(f"  ❌ Data gap: {gap_days} days ({gap_months:.1f} months)")
            print(f"    Last data: {latest_dataset['end_date']}")
            print(f"    Current date: {current_date.strftime('%Y-%m-%d')}")
            
            # Suggest new period
            start_date = (latest_end + timedelta(days=1)).strftime("%Y_%m_%d")
            
            # Suggest ending at last month end to avoid partial month
            if current_date.day < 15:  # If early in month, end at previous month
                end_date = (current_date.replace(day=1) - timedelta(days=1)).strftime("%Y_%m_%d")
            else:  # Otherwise end at current month (or previous if we want complete months)
                end_date = current_date.replace(day=1).strftime("%Y_%m_%d")[:-3] + "_01"
                end_date = (datetime.strptime(end_date, "%Y_%m_%d") - timedelta(days=1)).strftime("%Y_%m_%d")
            
            print(f"    Suggested new period: {start_date}-{end_date}")
            print(f"    Command: python main.py --data-only")
            print(f"    Note: Update config.py END_DATE before fetching")