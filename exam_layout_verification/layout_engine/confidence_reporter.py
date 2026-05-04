"""
Confidence Reporter + Visual Overlay Generator

Takes results from:
  - ScanQualityAnalyzer
  - MarkingDetector  
  - ExamLayoutParser's TransformContext (anchor info)

Produces:
  1. Confidence report (JSON)
  2. Annotated image with color-coded confidence overlay
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .scan_quality_analyzer import ScanQualityReport
from .marking_detector import QuestionMarking

logger = logging.getLogger(__name__)


# ── Color scheme for confidence levels ────────────────────────────────────────
# All BGR

def confidence_color(conf: float) -> Tuple[int, int, int]:
    """
    Returns BGR color for confidence level:
      >= 0.80 → Green  (good)
      >= 0.55 → Orange (warning)
      <  0.55 → Red    (fail)
    """
    if conf >= 0.80:
        return (0, 200, 0)      # Green
    elif conf >= 0.55:
        return (0, 140, 255)    # Orange
    else:
        return (0, 0, 220)      # Red


def status_color(status: str) -> Tuple[int, int, int]:
    colors = {
        "GOOD": (0, 200, 0),
        "WARN": (0, 140, 255),
        "FAIL": (0, 0, 220),
        "PASS": (0, 200, 0),
        "PARTIAL": (0, 140, 255),
    }
    return colors.get(status, (128, 128, 128))


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FullReport:
    exam_id: str
    image_path: str
    template_path: str

    # Scores
    scan_quality_score: float
    alignment_score: float
    marking_score: float
    overall_confidence: float
    overall_status: str         # PASS / PARTIAL / FAIL

    # Sub-reports
    scan_quality: dict
    anchors_found: int
    anchors_total: int
    rotation_deg: float
    shift_dx: float
    shift_dy: float

    # Per-question
    question_markings: Dict[int, dict]
    total_warnings: List[str]

    # Summary
    total_questions: int
    questions_marked: int
    questions_double_marked: int
    questions_unmarked: int

    def to_dict(self) -> dict:
        return {
            "exam_id": self.exam_id,
            "image_path": self.image_path,
            "template_path": self.template_path,
            "scores": {
                "scan_quality": round(self.scan_quality_score, 4),
                "alignment": round(self.alignment_score, 4),
                "marking_clarity": round(self.marking_score, 4),
                "overall_confidence": round(self.overall_confidence, 4),
                "overall_status": self.overall_status
            },
            "scan_analysis": self.scan_quality,
            "alignment": {
                "anchors_found": self.anchors_found,
                "anchors_total": self.anchors_total,
                "rotation_deg": round(self.rotation_deg, 3),
                "shift_dx": round(self.shift_dx, 2),
                "shift_dy": round(self.shift_dy, 2)
            },
            "summary": {
                "total_questions": self.total_questions,
                "questions_marked": self.questions_marked,
                "questions_double_marked": self.questions_double_marked,
                "questions_unmarked": self.questions_unmarked
            },
            "warnings": self.total_warnings,
            "questions": self.question_markings
        }


# ── Main reporter ─────────────────────────────────────────────────────────────

class ConfidenceReporter:
    """
    Aggregates all analysis results and generates:
    1. Full JSON report
    2. Color-coded visual overlay
    """

    def build_report(
        self,
        exam_id: str,
        image_path: str,
        template_path: str,
        scan_quality: ScanQualityReport,
        markings: Dict[int, QuestionMarking],
        anchors_found: int,
        anchors_total: int,
        ctx  # TransformContext
    ) -> FullReport:
        """Build the full confidence report."""

        # 1. Scan quality score (already computed)
        sq_score = scan_quality.overall_score

        # 2. Alignment score
        found_ratio = anchors_found / max(anchors_total, 1)
        rotation_penalty = min(1.0, abs(scan_quality.rotation_deg) / 10.0)
        alignment_score = found_ratio * (1.0 - rotation_penalty * 0.5)

        # 3. Marking score
        if markings:
            marking_confs = [m.marking_confidence for m in markings.values()]
            marking_score = float(np.mean(marking_confs))
        else:
            marking_score = 0.0

        # 4. Overall confidence (weighted)
        overall = (
            0.25 * sq_score
            + 0.25 * alignment_score
            + 0.50 * marking_score
        )
        overall = round(min(1.0, max(0.0, overall)), 4)

        # 5. Status
        all_warnings: List[str] = list(scan_quality.warnings)
        for m in markings.values():
            all_warnings.extend(m.warnings)

        critical_scan = scan_quality.overall_status == "FAIL"
        if critical_scan or overall < 0.45:
            status = "FAIL"
        elif overall < 0.72:
            status = "PARTIAL"
        else:
            status = "PASS"

        # 6. Summary stats
        total_q = len(markings)
        marked = sum(
            1 for m in markings.values()
            if not m.is_unmarked
        )
        double = sum(
            1 for m in markings.values()
            if m.is_double_marked
        )
        unmarked = sum(
            1 for m in markings.values()
            if m.is_unmarked
        )

        # Shift from ctx
        shift_dx = getattr(ctx, 'offset_x', 0.0)
        shift_dy = getattr(ctx, 'offset_y', 0.0)

        return FullReport(
            exam_id=exam_id,
            image_path=image_path,
            template_path=template_path,
            scan_quality_score=sq_score,
            alignment_score=alignment_score,
            marking_score=marking_score,
            overall_confidence=overall,
            overall_status=status,
            scan_quality=scan_quality.to_dict(),
            anchors_found=anchors_found,
            anchors_total=anchors_total,
            rotation_deg=scan_quality.rotation_deg,
            shift_dx=shift_dx,
            shift_dy=shift_dy,
            question_markings={
                q_num: m.to_dict()
                for q_num, m in markings.items()
            },
            total_warnings=all_warnings,
            total_questions=total_q,
            questions_marked=marked,
            questions_double_marked=double,
            questions_unmarked=unmarked
        )

    def draw_confidence_overlay(
        self,
        image: np.ndarray,
        report: FullReport,
        page_data: dict,
        ctx,
        markings: Dict[int, QuestionMarking],
        scan_quality: ScanQualityReport
    ) -> np.ndarray:
        """
        Draw confidence overlay on top of the annotated image.

        Shows:
        - Color-coded question boxes (green/orange/red)
        - Marking indicators (✓ / ✗ / ?)
        - Overall confidence badge
        - Scan quality badge
        - Per-question confidence scores
        """
        overlay = image.copy()
        img_h, img_w = overlay.shape[:2]
        questions = page_data.get("questions", {})

        # ── Draw per-question confidence boxes ──────────────────────────────
        for q_num_str, q_data in questions.items():
            q_num = int(q_num_str)
            marking = markings.get(q_num)
            if marking is None:
                continue

            bb = q_data.get("boundingBox", {})
            if not bb:
                continue

            conf = marking.marking_confidence
            color = confidence_color(conf)

            # Get polygon corners
            pts = ctx.apply_rect_vertices(
                bb["x"], bb["y"], bb["w"], bb["h"]
            )

            # Draw thick colored border
            cv2.polylines(
                overlay, [pts], isClosed=True,
                color=color, thickness=4, lineType=cv2.LINE_AA
            )

            # Semi-transparent fill
            fill_layer = overlay.copy()
            cv2.fillPoly(fill_layer, [pts], color)
            cv2.addWeighted(fill_layer, 0.08, overlay, 0.92, 0, overlay)

            # Confidence badge on question
            top_left = tuple(pts[0][0])
            badge_x, badge_y = top_left[0], top_left[1] - 2

            # Marking status icon
            if marking.is_double_marked:
                icon = "!!"
                icon_color = (0, 0, 220)
            elif marking.is_unmarked:
                icon = "?"
                icon_color = (0, 140, 255)
            elif conf >= 0.80:
                icon = "OK"
                icon_color = (0, 180, 0)
            else:
                icon = "~"
                icon_color = (0, 100, 200)

            # Draw confidence label
            scale = img_w / 756
            font_scale = max(0.35, 0.45 * scale)
            label = f"Q{q_num} {conf:.0%} {icon}"
            self._draw_badge(overlay, label, badge_x, badge_y, color)

            # Draw marking indicator for MC options
            if marking.q_type in ("multiple_choice", "multi_select"):
                self._draw_option_indicators(
                    overlay, q_data, marking, ctx, scale
                )

        # ── Draw overall confidence panel (top-right corner) ────────────────
        self._draw_overall_panel(overlay, report, img_w, img_h)

        # ── Draw scan quality bar (top-left) ────────────────────────────────
        self._draw_scan_quality_bar(overlay, scan_quality, img_h)

        return overlay

    def _draw_option_indicators(
        self,
        image: np.ndarray,
        q_data: dict,
        marking: QuestionMarking,
        ctx,
        scale: float
    ) -> None:
        """Draw ✓ or circle indicator next to each option."""
        options = q_data.get("options", {})

        for label, opt_box in options.items():
            om = next(
                (o for o in marking.option_markings if o.label == label),
                None
            )
            if om is None:
                continue

            # Get center of option box
            cx = opt_box["x"] + opt_box["w"] / 2
            cy = opt_box["y"] + opt_box["h"] / 2
            px, py = ctx.apply_point(cx, cy)

            r = max(5, int(8 * scale))

            if om.is_clearly_marked:
                # Filled circle = selected
                cv2.circle(image, (px, py), r, (0, 180, 0), -1)
                cv2.circle(image, (px, py), r, (0, 100, 0), 1)
            elif om.is_marked:
                # Partial fill = ambiguous
                cv2.circle(image, (px, py), r, (0, 140, 255), 2)
                cv2.circle(image, (px, py), r // 2, (0, 140, 255), -1)
            else:
                # Empty circle = not selected (faint)
                cv2.circle(image, (px, py), r, (180, 180, 180), 1)

    def _draw_badge(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw a small colored badge with text."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = image.shape[1] / 2000
        font_scale = max(0.3, 0.4 * image.shape[1] / 756)
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Clamp to image bounds
        x = max(0, min(x, image.shape[1] - tw - 4))
        y = max(th + 4, min(y, image.shape[0] - 4))

        # Background
        cv2.rectangle(
            image,
            (x - 2, y - th - baseline - 2),
            (x + tw + 2, y + baseline),
            color, cv2.FILLED
        )
        # Text
        cv2.putText(
            image, text, (x, y - baseline),
            font, font_scale, (255, 255, 255),
            thickness, cv2.LINE_AA
        )

    def _draw_overall_panel(
        self,
        image: np.ndarray,
        report: FullReport,
        img_w: int,
        img_h: int
    ) -> None:
        """Draw overall confidence panel in top-right area."""
        panel_w = max(280, int(img_w * 0.25))
        panel_h = 180
        panel_x = img_w - panel_w - 10
        panel_y = 10

        # Background
        overlay_bg = image.copy()
        cv2.rectangle(
            overlay_bg,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (30, 30, 30), cv2.FILLED
        )
        cv2.addWeighted(overlay_bg, 0.85, image, 0.15, 0, image)

        # Border
        s_color = status_color(report.overall_status)
        cv2.rectangle(
            image,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            s_color, 3
        )

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.4, img_w / 2500)
        tx = panel_x + 10
        ty = panel_y + 25

        def put(text, color=(255, 255, 255), bold=False):
            nonlocal ty
            cv2.putText(
                image, text, (tx, ty), font,
                fs * (1.3 if bold else 1.0),
                color, 2 if bold else 1, cv2.LINE_AA
            )
            ty += int(28 * fs * 2.5)

        conf = report.overall_confidence
        put(f"OVERALL: {report.overall_status}", s_color, bold=True)
        put(f"Confidence: {conf:.1%}", confidence_color(conf))
        put(f"Scan:  {report.scan_quality_score:.1%}", (200, 200, 200))
        put(f"Align: {report.alignment_score:.1%}", (200, 200, 200))
        put(f"Mark:  {report.marking_score:.1%}", (200, 200, 200))

        # Warnings count
        n_warn = len(report.total_warnings)
        warn_color = (0, 0, 220) if n_warn > 0 else (0, 180, 0)
        put(f"Warnings: {n_warn}", warn_color)

    def _draw_scan_quality_bar(
        self,
        image: np.ndarray,
        scan_quality: ScanQualityReport,
        img_h: int
    ) -> None:
        """Draw scan quality indicators on the left side."""
        x = 10
        y = img_h - 10

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.35, image.shape[1] / 3000)

        lines = [
            f"Scan: {scan_quality.overall_status} ({scan_quality.overall_score:.0%})",
            f"Blur: {next((c.value for c in scan_quality.checks if c.name=='blur'), 0):.0f}",
            f"Rot: {scan_quality.rotation_deg:+.1f}deg",
        ]

        for line in reversed(lines):
            (tw, th), _ = cv2.getTextSize(line, font, fs, 1)
            cv2.rectangle(
                image, (x - 2, y - th - 4), (x + tw + 2, y + 2),
                (30, 30, 30), cv2.FILLED
            )
            sq_color = status_color(scan_quality.overall_status)
            cv2.putText(
                image, line, (x, y),
                font, fs, sq_color, 1, cv2.LINE_AA
            )
            y -= th + 8