"""
Run exam layout verification.

Usage:
    python run_validation.py --image examples/ICE101-midterm-2026.png --template templates/ICE101_midterm.json
    python run_validation.py --image examples/ICE101-midterm-2026.png --template templates/ICE101_midterm.json --debug
"""

import argparse
import json
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def _conf_bar(conf: float, width: int = 10) -> str:
    filled = int(conf * width)
    return "█" * filled + "░" * (width - filled)


def _status_emoji(status: str) -> str:
    return {"PASS": "✅", "PARTIAL": "⚠️ ", "FAIL": "❌"}.get(status, "❓")


def print_report(report: dict):
    """Pretty-print the verification report to terminal."""

    status = report["overall_status"]
    conf   = report["overall_confidence"]
    emoji  = _status_emoji(status)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  EXAM LAYOUT VERIFICATION REPORT")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Exam:       {report['exam_id']}")
    print(f"  Type:       {report['exam_type'].upper()}  |  Date: {report['exam_date']}")
    print(f"  Image:      {report['image_path']}")
    print(f"  Scale:      {report['scale']}x")
    print(f"  Time:       {report['processing_time_ms']:.0f}ms")
    print()

    print(f"  ┌─ RESULT {'─'*48}")
    print(f"  │  Status:     {emoji} {status}")
    print(f"  │  Confidence: {_conf_bar(conf)} {conf:.1%}")
    print(f"  └{'─'*57}")
    print()

    # Anchors
    anchors = report["anchors"]
    found   = anchors["found_count"]
    symbols = ""
    for pos in ["TL", "TR", "BL", "BR"]:
        r = anchors["details"][pos]
        symbols += "●" if r["found"] else "○"
    print(f"  ANCHORS  [{symbols}]  {found}/4 found  score={anchors['overall_score']:.1%}")

    for pos in ["TL", "TR", "BL", "BR"]:
        r = anchors["details"][pos]
        if r["found"]:
            print(f"    {pos}: ✓  dev={r['deviation_px']:.1f}px  "
                  f"conf={r['confidence']:.2f}  method={r['method']}")
        else:
            print(f"    {pos}: ✗  NOT FOUND")
    print()

    # Transform
    tf = report["transform"]
    tf_ok = "✅" if tf["is_valid"] else "⚠️ "
    print(f"  TRANSFORM  {tf_ok}  method={tf['method']}  "
          f"err={tf['reprojection_error_px']:.2f}px  "
          f"rot={tf['rotation_deg']:.2f}°")
    dx = tf['shift_vector']['dx']
    dy = tf['shift_vector']['dy']
    print(f"    Shift: dx={dx:+.1f}px  dy={dy:+.1f}px")
    print(f"    Anchors used: {tf['anchors_used']}")
    print()

    # Confidence breakdown
    cb = report["confidence_breakdown"]["breakdown"]
    print(f"  CONFIDENCE BREAKDOWN")
    print(f"    Anchors    {cb['anchor_score']:.1%} × {cb['anchor_weight']:.0%}")
    print(f"    Transform  {cb['transform_score']:.1%} × {cb['transform_weight']:.0%}")
    print(f"    Questions  {cb['question_score']:.1%} × {cb['question_weight']:.0%}")
    print()

    # Questions
    summary = report["summary"]
    print(f"  QUESTIONS  "
          f"✅ {summary['questions_ok']} ok  "
          f"⚠️  {summary['questions_warning']} warning  "
          f"❌ {summary['questions_fail']} fail  "
          f"({summary['total_questions']} total)")
    print()

    print(f"  {'Q':>3}  {'Type':<16}  {'Confidence':>10}  {'Bar':<12}  "
          f"{'BB Status':<10}  Issues")
    print(f"  {'─'*70}")

    for q_num_str in sorted(report["questions"].keys(), key=int):
        qr = report["questions"][q_num_str]
        q_conf   = qr["question_confidence"]
        q_type   = qr["q_type"]
        bb_status = qr["bounding_box"]["status"]
        n_issues = len(qr["issues"])

        issue_str = ""
        if n_issues > 0:
            issue_str = f"🔴 {n_issues}" if any(
                i.get("severity") == "CRITICAL" for i in qr["issues"]
            ) else f"🟡 {n_issues}"

        print(
            f"  {q_num_str:>3}  {q_type:<16}  "
            f"{q_conf:>9.1%}  {_conf_bar(q_conf, 10):<12}  "
            f"{bb_status:<10}  {issue_str}"
        )

    print()
    print(f"  📁 Report: report.json")
    if "--debug" in sys.argv:
        print(f"  🖼️  Debug:  debug_output/")
    print()


def main():
    parser = argparse.ArgumentParser(description="Exam Layout Verifier")
    parser.add_argument("--image",    required=True,  help="Scanned exam PNG")
    parser.add_argument("--template", required=True,  help="Template JSON")
    parser.add_argument("--output",   default="report.json", help="Report output path")
    parser.add_argument("--debug",    action="store_true",   help="Save debug image")
    parser.add_argument("--tolerance", type=float, default=10.0,
                        help="Position tolerance in px (default=10)")
    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    if not Path(args.template).exists():
        print(f"❌ Template not found: {args.template}")
        sys.exit(1)

    # Run
    from layout_engine.layout_validator import LayoutValidator

    validator = LayoutValidator(
        tolerance_px=args.tolerance,
        debug=args.debug
    )

    print(f"\n🔍 Verifying: {args.image}")
    print(f"📋 Template:  {args.template}\n")

    try:
        report = validator.validate(
            image_path=args.image,
            template_path=args.template
        )
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print and save
    print_report(report)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Report saved to: {args.output}")

    return 0 if report["overall_status"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())