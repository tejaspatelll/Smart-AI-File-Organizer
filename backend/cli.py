"""Command-line interface for Smart File Organizer (Enhanced).

Usage:
    python -m backend.cli --path /path/to/folder --apply
    python -m backend.cli --path /path/to/folder --undo
    python -m backend.cli --path /path/to/folder --summary

If --apply is omitted, a dry-run JSON plan is printed instead of moving files.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from backend.organizer import DirectoryScanner, EnhancedDirectoryScanner, GeminiClassifier, FileOrganizer, CustomPromptClassifier, IntelligentAIClassifier, EnhancedFileInfo, FileInfo
from backend.ai_providers import ai_system
from backend.config import config
from backend.SmartPlanner import SmartPlanner
from backend.models import MovePlanItem

# load_dotenv()

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


def _build_structure_summary(root: Path, max_depth: int = 4, max_entries: int = 10000) -> dict:
    """Recursively summarize the selected directory for AI context.

    Includes folder graph, file counts, and extension distribution.
    Bounded by max_entries to protect against huge trees.
    """
    root = root.expanduser().resolve()
    summary: Dict[str, Any] = {"root": str(root), "folders": {}, "total_files": 0, "ext_counts": {}}
    entries = 0

    for dirpath, dirnames, filenames in os.walk(root):
        try:
            rel_dir = str(Path(dirpath).relative_to(root)) or "."
        except Exception:
            rel_dir = "."
        if rel_dir not in summary["folders"]:
            summary["folders"][rel_dir] = {"subfolders": [], "files": 0, "ext_counts": {}}

        # Record subfolders (relative)
        for d in dirnames:
            sub_rel = str(Path(rel_dir) / d) if rel_dir != "." else d
            summary["folders"][rel_dir]["subfolders"].append(sub_rel)

        # Record files and extension counts
        for fn in filenames:
            ext = (Path(fn).suffix or "").lower()
            summary["folders"][rel_dir]["files"] += 1
            summary["folders"][rel_dir]["ext_counts"][ext] = summary["folders"][rel_dir]["ext_counts"].get(ext, 0) + 1
            summary["total_files"] += 1
            summary["ext_counts"][ext] = summary["ext_counts"].get(ext, 0) + 1

            entries += 1
            if entries >= max_entries:
                return summary

    return summary


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
    parser.add_argument("--intelligent", action="store_true", help="Use intelligent AI classifier with confidence-based decisions and folder preservation")
    parser.add_argument("--policies", type=str, help="Path to JSON file with organization policies")
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
    progress_reporter.report(0, 1, "Starting enhanced file scan...", "scan")

    # Use EnhancedDirectoryScanner for better performance and features
    scanner = EnhancedDirectoryScanner(root_path, max_workers=4)
    
    # Convert to async scanning if available, otherwise fall back to sync scan
    try:
        import asyncio
        files = asyncio.run(scanner.scan_async(progress_callback=lambda c, t, m: progress_reporter.report(c, t, m, "scan")))
        # Keep EnhancedFileInfo objects - no conversion needed for better features
        # files = [FileInfo(path=f.path, metadata=f.metadata, category=f.category, suggestion=f.suggestion) for f in files]
        print(f"✅ Enhanced scanning completed: {len(files)} files processed", file=sys.stderr)
    except Exception as e:
        print(f"Enhanced scanning failed ({e}), falling back to legacy scanner", file=sys.stderr)
        # Fallback to legacy scanner
        legacy_scanner = DirectoryScanner(root_path)
        files = legacy_scanner.scan(progress_callback=lambda c, t, m: progress_reporter.report(c, t, m, "scan"))
        scanner = legacy_scanner  # For compatibility with duplicate detection

    # Unified provider path (preferred)
    try:
        progress_reporter.report(1, 1, "Starting AI classification...", "classify")
        import asyncio
        # Initialize providers once per run
        asyncio.run(ai_system.initialize())

        # Build context for providers
        structure = _build_structure_summary(root_path, max_depth=4)
        context: Dict[str, Any] = {"existing_structure": structure, "metadata_only": True, "confidence_threshold": config.ai.confidence_threshold}
        if args.prompt:
            context["user_prompt"] = args.prompt
            print(f"🎯 Using custom prompt: '{args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}'", file=sys.stderr)
        if args.template:
            context["template"] = args.template
            print(f"📋 Using template: {args.template}", file=sys.stderr)
        if args.policies:
            try:
                with open(Path(args.policies).expanduser().resolve(), 'r', encoding='utf-8') as pf:
                    context["policies"] = json.load(pf)
                print("🛡️  Policies loaded and applied.", file=sys.stderr)
            except Exception as pe:
                print(f"Warning: Failed to load policies file: {pe}", file=sys.stderr)

        # Convert EnhancedFileInfo/FileInfo to models.FileInfo
        from backend.models import FileInfo as ModelFileInfo, FileMetadata
        model_files = []
        for f in files:
            meta = FileMetadata(
                name=f.metadata.get("name", f.path.name),
                extension=f.metadata.get("extension", ""),
                size_bytes=f.metadata.get("size", f.metadata.get("size_bytes", 0)),
                created_timestamp=f.metadata.get("created_ts", f.metadata.get("created", 0) or 0),
                modified_timestamp=f.metadata.get("modified_ts", f.metadata.get("modified", 0) or 0),
                path=f.path,
            )
            model_files.append(ModelFileInfo(path=f.path, metadata=meta))

        # Classify via unified providers (batched with structure context)
        result = asyncio.run(ai_system.classify_files(model_files, context))

        # Map results back into original FileInfo objects for planner compatibility
        ai_map = {r.file_path.name: r for r in result.results}
        for f in files:
            r = ai_map.get(f.path.name)
            if not r:
                continue
            f.category = r.category
            f.suggestion = r.suggestion
            f.metadata["ai_confidence"] = r.confidence
            f.metadata["ai_reasoning"] = r.reasoning
            f.metadata["ai_priority"] = r.priority
            f.metadata["ai_tags"] = r.tags
        print(f"🤖 AI provider: {result.provider.value if hasattr(result, 'provider') else 'multi'} | files: {len(result.results)}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Unified AI classification failed: {e}. Falling back to legacy path.", file=sys.stderr)
        # Legacy fallback path retains previous behavior
        classifier = None
        if os.getenv("GEMINI_API_KEY"):
            try:
                # Use IntelligentAIClassifier if --intelligent flag is specified
                if args.intelligent:
                    classifier = IntelligentAIClassifier()
                    print("🧠 Using Intelligent AI Classifier with confidence-based decisions and folder preservation", file=sys.stderr)
                    files = classifier.classify_with_intelligence(files, {}, progress_callback=lambda c, t, m: progress_reporter.report(c, t, m, "classify"))
                # Use CustomPromptClassifier if prompt or template specified
                elif args.prompt or args.template:
                    classifier = CustomPromptClassifier()
                    if args.prompt:
                        classifier.set_custom_prompt(args.prompt)
                    if args.template:
                        classifier.set_organization_template(args.template)
                    files = classifier.classify_batch_with_prompt(files, progress_callback=lambda c, t, m: progress_reporter.report(c, t, m, "classify"))
                else:
                    classifier = GeminiClassifier()
                    files = classifier.classify_batch(files, progress_callback=lambda c, t, m: progress_reporter.report(c, t, m, "classify"))
            except Exception as e2:
                print(f"Warning: AI Classification failed: {e2}", file=sys.stderr)
    
    # Use SmartPlanner to ensure consistent AI-driven moves
    try:
        planner = SmartPlanner(root_path)
        # Convert to models.FileInfo list if not already
        from backend.models import FileInfo as ModelFileInfo, FileMetadata
        model_files = []
        for f in files:
            if isinstance(f, ModelFileInfo):
                model_files.append(f)
            else:
                meta = FileMetadata(
                    name=getattr(f, 'metadata', {}).get('name', f.path.name) if isinstance(getattr(f, 'metadata', {}), dict) else getattr(f.metadata, 'name', f.path.name),
                    extension=getattr(f, 'metadata', {}).get('extension', '') if isinstance(getattr(f, 'metadata', {}), dict) else getattr(f.metadata, 'extension', ''),
                    size_bytes=getattr(f, 'metadata', {}).get('size', getattr(getattr(f, 'metadata', {}), 'size_bytes', 0)) if isinstance(getattr(f, 'metadata', {}), dict) else getattr(f.metadata, 'size', getattr(f.metadata, 'size_bytes', 0)),
                    created_timestamp=getattr(f, 'metadata', {}).get('created_ts', getattr(getattr(f, 'metadata', {}), 'created_timestamp', 0)) if isinstance(getattr(f, 'metadata', {}), dict) else getattr(f.metadata, 'created_ts', getattr(f.metadata, 'created_timestamp', 0)),
                    modified_timestamp=getattr(f, 'metadata', {}).get('modified_ts', getattr(getattr(f, 'metadata', {}), 'modified_timestamp', 0)) if isinstance(getattr(f, 'metadata', {}), dict) else getattr(f.metadata, 'modified_ts', getattr(f.metadata, 'modified_timestamp', 0)),
                    path=f.path,
                )
                mf = ModelFileInfo(path=f.path, metadata=meta)
                # carry over AI results if present
                if getattr(f, 'category', None):
                    mf.set_category(getattr(f, 'category'))
                mf.suggestion = getattr(f, 'suggestion', None)
                mf.confidence = float(getattr(f, 'metadata', {}).get('ai_confidence', 0.0)) if isinstance(getattr(f, 'metadata', {}), dict) else float(getattr(getattr(f, 'metadata', {}), 'ai_confidence', 0.0))
                model_files.append(mf)

        org_plan = planner.build_plan(model_files)
        # Keep duplicate handling if requested by merging plans
        if args.include_duplicates:
            # Build organizer plan for duplicates
            organizer.build_plan(files, scanner)
            # Merge: keep duplicates from organizer into org_plan
            for item in organizer.plan:
                if item.is_duplicate:
                    org_plan.add_move(MovePlanItem(
                        source=item.source,
                        destination=item.destination,
                        priority=item.priority,
                        confidence=item.confidence,
                        reason=item.reason,
                    ))

        # Export frontend compatible array
        plan_array = SmartPlanner.to_frontend_array(org_plan)
    except Exception:
        # Fallback to legacy planner
        if args.include_duplicates:
            organizer.build_plan(files, scanner)
        else:
            organizer.build_plan(files)
        plan_array = json.loads(organizer.export_plan_json())

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
        print(json.dumps(plan_array, indent=2))


if __name__ == "__main__":
    main() 