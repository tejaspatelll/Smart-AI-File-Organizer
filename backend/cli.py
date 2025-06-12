"""Command-line interface for Smart File Organizer (Enhanced).

Usage:
    python -m backend.cli --path /path/to/folder --apply
    python -m backend.cli --path /path/to/folder --undo
    python -m backend.cli --path /path/to/folder --summary

If --apply is omitted, a dry-run JSON plan is printed instead of moving files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from .organizer import DirectoryScanner, GeminiClassifier, FileOrganizer, CustomPromptClassifier

class ProgressReporter:
    """A simple class to report progress back to the Electron app via stderr."""
    def __init__(self, total_steps: int | None = None) -> None:
        self.total_steps = total_steps
        self.current_step = 0
    
    def report(self, current: int, total: int, message: str, stage: str | None = None) -> None:
        # Calculate overall progress if total_steps is known
        overall_progress = 0
        if self.total_steps and stage == "scan":
            overall_progress = int((current / total) * (100 / self.total_steps))
        elif self.total_steps and stage == "classify":
            # Assuming scanning is step 1, classification is step 2
            overall_progress = int((100 / self.total_steps) + (current / total) * (100 / self.total_steps))
        else:
            overall_progress = int((current / total) * 100) # Fallback to 100% if single stage
            
        progress_data = {
            "type": "progress",
            "current": current,
            "total": total,
            "percentage": overall_progress,
            "message": message,
            "stage": stage
        }
        print(json.dumps(progress_data), file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart File Organizer Backend")
    parser.add_argument("--path", required=True, help="Directory to organize")
    parser.add_argument("--apply", action="store_true", help="Apply a given plan from stdin")
    parser.add_argument("--undo", action="store_true", help="Generate undo plan for recent moves")
    parser.add_argument("--summary", action="store_true", help="Show plan summary only")
    parser.add_argument("--include-duplicates", action="store_true", help="Include duplicate detection")
    parser.add_argument("--limit", type=int, default=50, help="Limit for undo operations")
    parser.add_argument("--prompt", type=str, help="Custom organization prompt")
    parser.add_argument("--template", type=str, choices=["creative", "business", "student", "personal"], help="Organization template")
    args = parser.parse_args()

    root_path = Path(args.path).expanduser().resolve()
    organizer = FileOrganizer(root_path)

    if args.apply:
        try:
            items_to_apply = json.load(sys.stdin)
            organizer.apply_plan(items_to_apply)
            print("✔️ Plan applied successfully.", file=sys.stderr)
        except json.JSONDecodeError:
            print("Error: Invalid JSON received for apply plan.", file=sys.stderr)
            sys.exit(1)
        return
    
    if args.undo:
        try:
            undo_plan = organizer.create_undo_plan(args.limit)
            if undo_plan:
                print(json.dumps(undo_plan, indent=2))
                print(f"✔️ Generated undo plan for {len(undo_plan)} operations.", file=sys.stderr)
            else:
                print("No recent moves to undo.", file=sys.stderr)
        except Exception as e:
            print(f"Error generating undo plan: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Plan generation ---
    # Report that we are starting the process
    progress_reporter = ProgressReporter(total_steps=2) # Scan and Classify are two main steps
    progress_reporter.report(0, 1, "Starting file scan...", "scan")

    scanner = DirectoryScanner(root_path)
    files = scanner.scan(progress_callback=lambda c, t, m: progress_reporter.report(c, t, m, "scan"))

    classifier = None
    if os.getenv("GEMINI_API_KEY"):
        try:
            progress_reporter.report(1, 1, "Starting AI classification...", "classify")
            
            # Use CustomPromptClassifier if prompt or template specified
            if args.prompt or args.template:
                classifier = CustomPromptClassifier()
                if args.prompt:
                    classifier.set_custom_prompt(args.prompt)
                    print(f"🎯 Using custom prompt: '{args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}'", file=sys.stderr)
                if args.template:
                    classifier.set_organization_template(args.template)
                    print(f"📋 Using template: {args.template}", file=sys.stderr)
                files = classifier.classify_batch_with_prompt(files, progress_callback=lambda c, t, m: progress_reporter.report(c, t, m, "classify"))
            else:
                classifier = GeminiClassifier()
                files = classifier.classify_batch(files, progress_callback=lambda c, t, m: progress_reporter.report(c, t, m, "classify"))
        except Exception as e:
            print(f"Warning: AI Classification failed: {e}", file=sys.stderr)
    
    # Build plan with duplicate detection if requested
    if args.include_duplicates:
        organizer.build_plan(files, scanner)
    else:
        organizer.build_plan(files)

    if args.summary:
        # Show only summary
        summary = organizer.get_plan_summary()
        print(json.dumps(summary, indent=2))
        
        # Print human-readable summary to stderr
        print(f"\n📊 Plan Summary:", file=sys.stderr)
        print(f"   Total files: {summary['total_files']}", file=sys.stderr)
        print(f"   Duplicates: {summary['duplicates']}", file=sys.stderr)
        print(f"   Average confidence: {summary['avg_confidence']:.2f}", file=sys.stderr)
        print(f"   High confidence moves: {summary['high_confidence_count']}", file=sys.stderr)
        
        if summary['categories']:
            print(f"   Categories:", file=sys.stderr)
            for category, count in summary['categories'].items():
                print(f"     - {category}: {count} files", file=sys.stderr)
    else:
        # Print full JSON plan to stdout
        print(organizer.export_plan_json())


if __name__ == "__main__":
    main() 