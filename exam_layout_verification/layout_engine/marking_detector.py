"""
Marking Detector - FIXED v2

Fixes:
  1. Warning message shows correct center_darkness value
  2. Multi-select confidence scoring improved
  3. Ambiguous detection threshold tuned
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
CIRCLE_MARKED_THRESHOLD = 0.25
CIRCLE_CLEARLY_MARKED   = 0.45
CIRCLE_AMBIGUOUS_LOW    = 0.15
BOX_HAS_TEXT_THRESHOLD  = 0.03


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class OptionMarking:
    label: str
    raw_darkness: float
    center_darkness: float
    is_marked: bool
    is_clearly_marked: bool
    is_ambiguous: bool

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "raw_darkness": round(self.raw_darkness, 4),
            "center_darkness": round(self.center_darkness, 4),
            "is_marked": self.is_marked,
            "is_clearly_marked": self.is_clearly_marked,
            "is_ambiguous": self.is_ambiguous
        }


@dataclass
class QuestionMarking:
    question_number: int
    q_type: str
    option_markings: List[OptionMarking] = field(default_factory=list)
    selected_options: List[str] = field(default_factory=list)
    is_double_marked: bool = False
    is_unmarked: bool = False
    box_has_content: Dict[str, bool] = field(default_factory=dict)
    box_darkness: Dict[str, float] = field(default_factory=dict)
    marking_confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question_number": self.question_number,
            "q_type": self.q_type,
            "selected_options": self.selected_options,
            "is_double_marked": self.is_double_marked,
            "is_unmarked": self.is_unmarked,
            "marking_confidence": round(self.marking_confidence, 4),
            "box_has_content": self.box_has_content,
            "box_darkness": {k: round(v, 4) for k, v in self.box_darkness.items()},
            "option_markings": [o.to_dict() for o in self.option_markings],
            "warnings": self.warnings
        }


# ── Main detector ─────────────────────────────────────────────────────────────

class MarkingDetector:

    def __init__(self, debug: bool = False):
        self.debug = debug

    def detect_all(
        self,
        image: np.ndarray,
        questions: Dict[str, dict],
        ctx
    ) -> Dict[int, "QuestionMarking"]:
        results = {}
        for q_num_str, q_data in questions.items():
            q_num  = int(q_num_str)
            q_type = q_data.get("type", "unknown")
            marking = self._detect_question(q_num, q_type, q_data, image, ctx)
            results[q_num] = marking
            self._log_marking(marking)
        return results

    def _detect_question(self, q_num, q_type, q_data, image, ctx):
        if q_type == "multiple_choice":
            return self._detect_mc(q_num, q_type, q_data, image, ctx, single=True)
        elif q_type == "multi_select":
            return self._detect_mc(q_num, q_type, q_data, image, ctx, single=False)
        elif q_type in ("matching", "fill_blanks"):
            return self._detect_boxes(q_num, q_type, q_data, image, ctx)
        elif q_type == "open_ended":
            return self._detect_open_ended(q_num, q_data, image, ctx)
        else:
            return QuestionMarking(
                question_number=q_num, q_type=q_type,
                is_unmarked=True, marking_confidence=0.5
            )

    def _detect_mc(self, q_num, q_type, q_data, image, ctx, single: bool):
        """
        Detect MC/multi-select markings using center darkness.
        """
        options = q_data.get("options", {})
        option_markings: List[OptionMarking] = []
        warnings: List[str] = []

        for label, opt_box in options.items():
            raw_darkness    = self._measure_darkness_full(image, opt_box, ctx)
            center_darkness = self._measure_darkness_center(
                image, opt_box, ctx, shrink_ratio=0.40
            )

            is_marked    = center_darkness > CIRCLE_MARKED_THRESHOLD
            is_clearly   = center_darkness > CIRCLE_CLEARLY_MARKED
            is_ambiguous = (
                CIRCLE_AMBIGUOUS_LOW < center_darkness <= CIRCLE_MARKED_THRESHOLD
            )

            option_markings.append(OptionMarking(
                label=label,
                raw_darkness=raw_darkness,
                center_darkness=center_darkness,
                is_marked=is_marked,
                is_clearly_marked=is_clearly,
                is_ambiguous=is_ambiguous
            ))

            logger.debug(
                f"    Q{q_num} opt {label}: "
                f"raw={raw_darkness:.3f} center={center_darkness:.3f} "
                f"{'MARKED' if is_marked else 'empty'}"
            )

        selected         = [om.label for om in option_markings if om.is_marked]
        clearly_selected = [om.label for om in option_markings if om.is_clearly_marked]
        ambiguous_opts   = [om for om in option_markings if om.is_ambiguous]

        is_double  = len(selected) > 1 if single else False
        is_unmarked = len(selected) == 0

        # ── Build warnings (FIXED: show actual option's darkness) ─────────────
        if is_double:
            warnings.append(
                f"⚠️  DOUBLE MARKED Q{q_num}: "
                f"options {selected} are both filled - select only ONE"
            )

        if is_unmarked:
            warnings.append(f"⚠️  UNMARKED: Q{q_num} has no selected option")

        # Fixed: get each ambiguous option's OWN darkness value
        for am_opt in ambiguous_opts:
            warnings.append(
                f"⚠️  UNCLEAR Q{q_num} option {am_opt.label}: "
                f"mark more completely "
                f"(center_darkness={am_opt.center_darkness:.2f})"
            )

        # ── Confidence scoring ────────────────────────────────────────────────
        if q_type == "multi_select":
            # Multi-select: confidence based on clarity of selected options
            if is_unmarked:
                confidence = 0.10
            elif all(om.is_clearly_marked for om in option_markings if om.is_marked):
                # All selected options are clearly marked
                confidence = 0.90
            elif ambiguous_opts:
                confidence = 0.60
            else:
                confidence = 0.75
        else:
            # Single choice
            if is_double:
                confidence = 0.30
            elif is_unmarked:
                confidence = 0.10
            elif len(clearly_selected) == 1 and not ambiguous_opts:
                confidence = 0.95
            elif len(selected) == 1:
                confidence = 0.70
            else:
                confidence = 0.50

        return QuestionMarking(
            question_number=q_num,
            q_type=q_type,
            option_markings=option_markings,
            selected_options=selected,
            is_double_marked=is_double,
            is_unmarked=is_unmarked,
            marking_confidence=confidence,
            warnings=warnings
        )

    def _detect_boxes(self, q_num, q_type, q_data, image, ctx):
        """Detect content in answer boxes."""
        warnings: List[str] = []
        box_has_content: Dict[str, bool] = {}
        box_darkness: Dict[str, float] = {}

        answer_boxes = q_data.get("answerBoxes", {})
        for box_id, box in answer_boxes.items():
            darkness    = self._measure_darkness_inner_box(image, box, ctx)
            has_content = darkness > BOX_HAS_TEXT_THRESHOLD
            box_has_content[str(box_id)] = has_content
            box_darkness[str(box_id)]    = darkness

            if not has_content:
                warnings.append(
                    f"⚠️  EMPTY BOX: Q{q_num} answer box {box_id} is empty"
                )

        filled = sum(1 for v in box_has_content.values() if v)
        total  = len(box_has_content)

        if total == 0:
            confidence = 0.5
        elif filled == total:
            confidence = 0.90
        elif filled == 0:
            confidence = 0.10
        else:
            confidence = 0.50 + 0.40 * (filled / total)

        return QuestionMarking(
            question_number=q_num,
            q_type=q_type,
            box_has_content=box_has_content,
            box_darkness=box_darkness,
            is_unmarked=(filled == 0),
            marking_confidence=confidence,
            warnings=warnings
        )

    def _detect_open_ended(self, q_num, q_data, image, ctx):
        """Detect writing in open-ended solution area."""
        warnings: List[str] = []
        solution_area = q_data.get("solutionArea")

        if not solution_area:
            return QuestionMarking(
                question_number=q_num, q_type="open_ended",
                is_unmarked=True, marking_confidence=0.5,
                warnings=["No solution area in template"]
            )

        darkness    = self._measure_writing_density(image, solution_area, ctx)
        has_content = darkness > 0.01

        if not has_content:
            warnings.append(f"⚠️  BLANK: Q{q_num} open-ended answer area is empty")
            confidence = 0.10
        else:
            confidence = min(0.95, 0.50 + darkness * 5)

        return QuestionMarking(
            question_number=q_num,
            q_type="open_ended",
            box_has_content={"solution_area": has_content},
            box_darkness={"solution_area": round(darkness, 4)},
            is_unmarked=not has_content,
            marking_confidence=confidence,
            warnings=warnings
        )

    # ── Measurement methods ───────────────────────────────────────────────────

    def _get_crop(
        self,
        image: np.ndarray,
        region: dict,
        ctx,
        shrink_ratio: float = 1.0
    ) -> Optional[np.ndarray]:
        img_h, img_w = image.shape[:2]

        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 10)
        h = region.get("h", 10)

        if shrink_ratio < 1.0:
            shrink_x = w * (1 - shrink_ratio) / 2
            shrink_y = h * (1 - shrink_ratio) / 2
            x += shrink_x
            y += shrink_y
            w *= shrink_ratio
            h *= shrink_ratio

        pts     = ctx.apply_rect_vertices(x, y, w, h)
        pts_arr = pts.reshape(-1, 2)

        x1 = max(0, int(pts_arr[:, 0].min()))
        y1 = max(0, int(pts_arr[:, 1].min()))
        x2 = min(img_w, int(pts_arr[:, 0].max()))
        y2 = min(img_h, int(pts_arr[:, 1].max()))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _measure_darkness_full(self, image, region, ctx) -> float:
        crop = self._get_crop(image, region, ctx, shrink_ratio=1.0)
        if crop is None:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return float(binary.mean()) / 255.0

    def _measure_darkness_center(
        self, image, region, ctx, shrink_ratio: float = 0.40
    ) -> float:
        crop = self._get_crop(image, region, ctx, shrink_ratio=shrink_ratio)
        if crop is None:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        return float(binary.mean()) / 255.0

    def _measure_darkness_inner_box(self, image, region, ctx) -> float:
        crop = self._get_crop(image, region, ctx, shrink_ratio=0.75)
        if crop is None:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        return float(binary.mean()) / 255.0

    def _measure_writing_density(self, image, region, ctx) -> float:
        crop = self._get_crop(image, region, ctx, shrink_ratio=0.90)
        if crop is None:
            return 0.0
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(edges.mean()) / 255.0

    def _log_marking(self, marking: QuestionMarking) -> None:
        if marking.q_type in ("multiple_choice", "multi_select"):
            sel    = marking.selected_options or ["(none)"]
            status = ("DOUBLE" if marking.is_double_marked
                      else "EMPTY" if marking.is_unmarked else "OK")
            logger.info(
                f"  Q{marking.question_number} ({marking.q_type}): "
                f"selected={sel} conf={marking.marking_confidence:.2f} [{status}]"
            )
        else:
            filled = sum(1 for v in marking.box_has_content.values() if v)
            total  = len(marking.box_has_content)
            logger.info(
                f"  Q{marking.question_number} ({marking.q_type}): "
                f"boxes {filled}/{total} filled "
                f"conf={marking.marking_confidence:.2f}"
            )