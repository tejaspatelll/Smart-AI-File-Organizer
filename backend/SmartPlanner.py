from __future__ import annotations

from pathlib import Path
from typing import List

from .models import OrganizationPlan, MovePlanItem, FileInfo, Priority
from .config import config
import os
from collections import defaultdict
from difflib import SequenceMatcher


class SmartPlanner:
    """Planner that converts classified FileInfo into OrganizationPlan and frontend-compatible array."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        # Build an index of existing folder names for reuse preference
        self._folder_name_index = defaultdict(list)
        for dirpath, dirnames, _ in os.walk(self.root):
            for d in dirnames:
                self._folder_name_index[d.lower()].append(Path(dirpath) / d)
        # Forbidden directories that should never appear in destination suggestions
        self._forbidden_dirs = {"node_modules", ".git", "build", "dist", "__pycache__", "venv", ".svn", ".hg"}

    def _category_name(self, category: object) -> str:
        """Return user-friendly category name (enum.value if enum)."""
        try:
            # Enum like FileCategory -> use .value
            return getattr(category, "value", str(category))
        except Exception:
            return str(category)

    def _sanitize_suggestion(self, suggestion: str) -> str:
        """Remove unsafe/forbidden path segments from suggestion."""
        try:
            parts = [seg for seg in Path(suggestion).parts if seg not in self._forbidden_dirs]
            safe = Path(*parts)
            return str(safe)
        except Exception:
            return suggestion

    def _normalize_suggestion(self, suggestion: str, filename: str) -> str:
        """Ensure suggestion represents a folder path, not a path including the filename.
        If the last segment matches the filename or its stem, drop it.
        Also remove forbidden segments (e.g., node_modules).
        """
        suggestion = self._sanitize_suggestion(suggestion)
        try:
            p = Path(suggestion)
            if not p.parts:
                return suggestion
            leaf = p.name
            stem = Path(filename).stem
            if leaf == filename or leaf == stem:
                p = p.parent
            # Also drop duplicate filename one level deeper (e.g., foo/bar.js/bar.js)
            if p.name == stem:
                # Keep as folder named stem if intentional
                pass
            return str(p) if str(p) != "." else ""
        except Exception:
            return suggestion

    def _map_to_existing(self, category: str, suggestion: str) -> Path:
        """Prefer existing subfolders close to the suggested path, constrained within category.
        1) Exact path under category exists
        2) Exact leaf folder name under category
        3) Fuzzy match within category
        4) Fallback: use suggested path under category
        """
        desired = self.root / category / suggestion
        try:
            if desired.exists():
                return desired
        except Exception:
            # If path is malformed, fallback to safe join
            desired = (self.root / category) / Path(suggestion)

        leaf = Path(suggestion).name.lower()
        cat_base = (self.root / category).resolve()

        # Exact leaf candidates strictly under category
        under_cat_candidates = []
        if leaf:
            for c in self._folder_name_index.get(leaf, []):
                try:
                    if str(c.resolve()).startswith(str(cat_base)):
                        under_cat_candidates.append(c)
                except Exception:
                    continue
            if under_cat_candidates:
                return under_cat_candidates[0]

        # Fuzzy search strictly under category
        best = None
        best_score = 0.0
        if (self.root / category).exists():
            for dirpath, dirnames, _ in os.walk(self.root / category):
                for d in dirnames:
                    score = SequenceMatcher(None, d.lower(), leaf).ratio() if leaf else 0.0
                    if score > best_score:
                        best_score = score
                        best = Path(dirpath) / d
        if best and best_score >= 0.82:
            return best

        return desired

    def build_plan(self, files: List[FileInfo]) -> OrganizationPlan:
        plan = OrganizationPlan()
        # Use configured threshold; default 0.7
        thr = getattr(config.ai, "confidence_threshold", 0.7)
        for f in files:
            if not getattr(f, 'category', None) or not getattr(f, 'suggestion', None):
                continue

            # get confidence from object or metadata
            try:
                confidence = float(getattr(f, 'confidence', 0.0) or getattr(getattr(f, 'metadata', {}), 'ai_confidence', 0.0))
            except Exception:
                confidence = float(getattr(f, 'confidence', 0.0) or 0.0)

            if confidence < thr:
                continue

            category_name = self._category_name(getattr(f, 'category', ""))
            normalized_suggestion = self._normalize_suggestion(str(getattr(f, 'suggestion', "")), f.path.name)

            dest_dir = self._map_to_existing(category_name, normalized_suggestion)
            dest = dest_dir / f.path.name
            try:
                # Ensure destination remains under root/category and has no forbidden segments
                if not str(dest_dir.resolve()).startswith(str((self.root / category_name).resolve())):
                    dest_dir = (self.root / category_name / normalized_suggestion)
                    dest = dest_dir / f.path.name
                if any(seg in self._forbidden_dirs for seg in dest.parts):
                    # Strip forbidden segments from the path
                    cleaned = [seg for seg in dest_dir.parts if seg not in self._forbidden_dirs]
                    dest_dir = Path(*cleaned)
                    dest = dest_dir / f.path.name
            except Exception:
                pass

            try:
                if dest.resolve() == f.path.resolve():
                    continue
            except Exception:
                # If resolve fails, still proceed with move plan as best effort
                pass

            move = MovePlanItem(
                source=f.path,
                destination=dest,
                priority=Priority(int(getattr(f, 'metadata', {}).get('ai_priority', 3)) if isinstance(getattr(f, 'metadata', {}), dict) else Priority(3)),
                confidence=confidence,
                reason=(getattr(getattr(f, 'metadata', {}), 'ai_reasoning', None) if not isinstance(getattr(f, 'metadata', {}), dict) else getattr(f, 'metadata', {}).get('ai_reasoning')) or 'AI suggestion'
            )
            plan.add_move(move)
        return plan

    @staticmethod
    def to_frontend_array(plan: OrganizationPlan) -> List[dict]:
        return [
            {
                "source": str(m.source),
                "destination": str(m.destination),
                "priority": int(m.priority),
                "confidence": float(m.confidence or 0.0),
                "reason": m.reason or "",
                "is_duplicate": bool(m.is_duplicate),
            }
            for m in plan.moves
        ]


