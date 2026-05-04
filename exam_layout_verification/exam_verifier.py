"""
Exam Verifier - Main Entry Point

Combines:
  - ExamLayoutParser (your existing code) for visualization
  - ScanQualityAnalyzer for blur/rotation/brightness
  - MarkingDetector for student answer detection
  - ConfidenceReporter for final scoring + overlay

Usage:
    python exam_verifier.py --image scan.png --template template.json
    python exam_verifier.py --image scan.png --template template.json --debug --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Import your existing ExamLayoutParser ─────────────────────────────────────
# Add parent to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from examlayoutparser import (
    ExamLayoutParser,
    TransformContext,
    compute_transform,
    Color
)

# ── Import new modules ────────────────────────────────────────────────────────
from layout_engine.scan_quality_analyzer import ScanQualityAnalyzer
from layout_engine.marking_detector import MarkingDetector
from layout_engine.confidence_reporter import ConfidenceReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Helper functions ──────────────────────────────────────────────────────────

def _bar(conf: float, width: int = 12) -> str:
    filled = int(conf * width)
    return "█" * filled + "░" * (width - filled)


def _status_icon(status: str) -> str:
    return {"PASS": "✅", "PARTIAL": "⚠️ ", "FAIL": "❌",
            "GOOD": "✅", "WARN": "⚠️ "}.get(status, "❓")


def print_terminal_report(report_dict: dict, markings: dict) -> None:
    """Print beautiful terminal output."""

    scores = report_dict["scores"]
    scan   = report_dict["scan_analysis"]
    align  = report_dict["alignment"]
    summ   = report_dict["summary"]
    status = scores["overall_status"]
    conf   = scores["overall_confidence"]

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         EXAM VERIFICATION REPORT                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Exam:     {report_dict['exam_id']}")
    print(f"  Image:    {report_dict['image_path']}")
    print()

    # ── Overall result ────────────────────────────────────────────────────────
    icon = _status_icon(status)
    print(f"  ┌─ RESULT {'─'*50}")
    print(f"  │  {icon} Status:     {status}")
    print(f"  │  📊 Confidence: {_bar(conf)} {conf:.1%}")
    print(f"  └{'─'*58}")
    print()

    # ── Score breakdown ───────────────────────────────────────────────────────
    print("  SCORE BREAKDOWN:")
    sq = scores["scan_quality"]
    al = scores["alignment"]
    mk = scores["marking_clarity"]
    print(f"    Scan Quality:  {_bar(sq, 10)} {sq:.1%}  (weight: 25%)")
    print(f"    Alignment:     {_bar(al, 10)} {al:.1%}  (weight: 25%)")
    print(f"    Marking Clarity:{_bar(mk, 10)} {mk:.1%}  (weight: 50%)")
    print()

    # ── Scan quality ──────────────────────────────────────────────────────────
    sq_status = scan["overall_status"]
    print(f"  SCAN QUALITY  {_status_icon(sq_status)} {sq_status}")
    for check in scan["checks"]:
        icon_c = _status_icon(check["status"])
        print(f"    {icon_c} {check['name']:<12} "
              f"{_bar(check['score'], 8)} {check['score']:.0%}  "
              f"→ {check['message']}")
    print()

    # ── Alignment ─────────────────────────────────────────────────────────────
    found = align["anchors_found"]
    total = align["anchors_total"]
    rot   = align["rotation_deg"]
    dx    = align["shift_dx"]
    dy    = align["shift_dy"]
    print(f"  ALIGNMENT")
    print(f"    Anchors:  {found}/{total} found  {'✅' if found==total else '⚠️ '}")
    print(f"    Rotation: {rot:+.2f}°  "
          f"{'✅ OK' if abs(rot) < 1.0 else '⚠️  slight' if abs(rot) < 5 else '❌ significant'}")
    print(f"    Shift:    dx={dx:+.1f}px  dy={dy:+.1f}px")
    print()

    # ── Questions ─────────────────────────────────────────────────────────────
    print(f"  QUESTIONS  "
          f"({summ['questions_marked']}/{summ['total_questions']} marked  "
          f"🔴 {summ['questions_double_marked']} double  "
          f"⬜ {summ['questions_unmarked']} empty)")
    print()
    print(f"  {'Q':>3}  {'Type':<16}  {'Confidence':>10}  {'Bar':<13}  "
          f"{'Selected':<12}  Status")
    print(f"  {'─'*75}")

    for q_num in sorted(report_dict["questions"].keys(), key=int):
        qm = report_dict["questions"][q_num]
        mk_conf  = qm["marking_confidence"]
        q_type   = qm["q_type"]
        selected = qm.get("selected_options", [])
        is_double = qm.get("is_double_marked", False)
        is_empty  = qm.get("is_unmarked", False)

        sel_str = ",".join(selected) if selected else "─"

        if is_double:
            status_str = "⚠️  DOUBLE MARK"
        elif is_empty:
            status_str = "⬜ EMPTY"
        elif mk_conf >= 0.80:
            status_str = "✅ Clear"
        elif mk_conf >= 0.55:
            status_str = "⚠️  Unclear"
        else:
            status_str = "❌ Problem"

        print(
            f"  {q_num:>3}  {q_type:<16}  "
            f"{mk_conf:>9.1%}  {_bar(mk_conf, 12):<13}  "
            f"{sel_str:<12}  {status_str}"
        )

    # ── Warnings ──────────────────────────────────────────────────────────────
    all_warnings = report_dict.get("warnings", [])
    if all_warnings:
        print()
        print(f"  WARNINGS ({len(all_warnings)} total):")
        for w in all_warnings:
            print(f"    {w}")

    print()


# ── Main verification function ────────────────────────────────────────────────

def verify_exam(
    image_path: str,
    template_path: str,
    output_dir: str = "output",
    debug: bool = False
) -> dict:
    """
    Full verification pipeline.

    Args:
        image_path:    Path to scanned exam PNG
        template_path: Path to template JSON
        output_dir:    Where to save outputs
        debug:         Save debug images

    Returns:
        Full report as dict
    """
    start = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_h, img_w = image.shape[:2]
    logger.info(f"Image: {img_w}×{img_h}px")

    # ── Load template ─────────────────────────────────────────────────────────
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    exam_id = template.get("examId", "unknown")
    page_data = template["pages"][0]
    ref_w = float(page_data["pageWidth"])
    ref_h = float(page_data["pageHeight"])
    scale_x = img_w / ref_w
    scale_y = img_h / ref_h
    anchors_def = page_data.get("anchors", {})

    logger.info(f"Template: {ref_w}×{ref_h}px, scale={scale_x:.3f}x")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Scan Quality Analysis
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Step 1: Analyzing scan quality...")
    analyzer = ScanQualityAnalyzer()
    scan_quality = analyzer.analyze(image)

    logger.info(
        f"  Scan: {scan_quality.overall_status} "
        f"(score={scan_quality.overall_score:.2f}, "
        f"blur={next((c.value for c in scan_quality.checks if c.name=='blur'), 0):.0f}, "
        f"rotation={scan_quality.rotation_deg:.2f}°)"
    )

    if not scan_quality.can_process:
        logger.error("Scan quality too poor to process reliably!")
        for w in scan_quality.warnings:
            logger.error(f"  {w}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Anchor Detection + Transform (using YOUR existing code)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Step 2: Computing geometric transform...")
    ctx = compute_transform(image, anchors_def, scale_x, scale_y)

    anchors_found = 0
    if ctx.homography is not None:
        anchors_found = 4  # Homography needs ≥4
    elif ctx.offset_x != 0 or ctx.offset_y != 0:
        anchors_found = 2  # Affine offset needs ≥2

    logger.info(
        f"  Transform: {'homography' if ctx.homography is not None else 'offset'} "
        f"({anchors_found}/{len(anchors_def)} anchors)"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Generate annotated image (YOUR existing ExamLayoutParser)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Step 3: Generating layout annotation...")

    # Use ExamLayoutParser to draw all regions
    annotated = image.copy()

    # Import drawing functions from your existing code
    from examlayoutparser import (
        ExamLayoutParser as ELP,
        draw_rect, draw_bullseye, draw_label
    )

    # Draw anchors
    ELP._draw_anchors(annotated, anchors_def, ctx)

    # Draw identity regions
    ELP._draw_identity_regions(annotated, page_data, ctx)

    # Draw questions
    questions = page_data.get("questions", {})
    ELP._draw_questions(annotated, questions, ctx)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Marking Detection
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Step 4: Detecting student markings...")
    marking_detector = MarkingDetector(debug=debug)
    markings = marking_detector.detect_all(image, questions, ctx)

    marked_count = sum(1 for m in markings.values() if not m.is_unmarked)
    double_count = sum(1 for m in markings.values() if m.is_double_marked)
    logger.info(
        f"  Markings: {marked_count}/{len(markings)} questions answered, "
        f"{double_count} double-marked"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Build Report + Confidence Overlay
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Step 5: Computing confidence scores...")
    reporter = ConfidenceReporter()

    full_report = reporter.build_report(
        exam_id=exam_id,
        image_path=image_path,
        template_path=template_path,
        scan_quality=scan_quality,
        markings=markings,
        anchors_found=anchors_found,
        anchors_total=len(anchors_def),
        ctx=ctx
    )

    # Draw confidence overlay on top of annotated image
    final_image = reporter.draw_confidence_overlay(
        image=annotated,
        report=full_report,
        page_data=page_data,
        ctx=ctx,
        markings=markings,
        scan_quality=scan_quality
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: Save Outputs
    # ══════════════════════════════════════════════════════════════════════════
    img_stem = Path(image_path).stem
    report_dict = full_report.to_dict()
    report_dict["processing_time_ms"] = round((time.time() - start) * 1000, 2)

    # Save annotated image
    img_out = out_dir / f"{img_stem}_verified.jpg"
    cv2.imwrite(str(img_out), final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info(f"  Annotated image: {img_out}")

    # Save JSON report
    json_out = out_dir / f"{img_stem}_report.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"  JSON report: {json_out}")

    print_terminal_report(report_dict, markings)

    print(f"  📁 Outputs saved to: {out_dir}/")
    print(f"  ⏱  Processing time: {report_dict['processing_time_ms']:.0f}ms")
    print()

    return report_dict


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exam Layout Verification with Confidence Scoring"
    )
    parser.add_argument("--image",    "-i", required=True,
                        help="Scanned exam PNG/JPG")
    parser.add_argument("--template", "-t", required=True,
                        help="Exam template JSON")
    parser.add_argument("--output",   "-o", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--debug",    "-d", action="store_true",
                        help="Enable debug mode")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    if not Path(args.template).exists():
        print(f"❌ Template not found: {args.template}")
        sys.exit(1)

    try:
        report = verify_exam(
            image_path=args.image,
            template_path=args.template,
            output_dir=args.output,
            debug=args.debug
        )
        status = report["scores"]["overall_status"]
        sys.exit(0 if status != "FAIL" else 1)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()