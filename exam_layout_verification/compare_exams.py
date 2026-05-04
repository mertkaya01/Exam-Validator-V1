"""
Compare multiple exam scans against the same template.
Perfect for your presentation: shows how different scans score differently.

Usage:
    python compare_exams.py --template templates/ICE101_midterm.json \
        --images examples/ICE101-midterm-2026.png \
                 examples/test_variants/ICE101-midterm-2026_rot_2deg.png \
                 examples/test_variants/ICE101-midterm-2026_student_marked.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2


def _bar(conf: float, w: int = 8) -> str:
    filled = int(conf * w)
    return "█" * filled + "░" * (w - filled)


def _icon(status: str) -> str:
    return {"PASS": "✅", "PARTIAL": "⚠️ ", "FAIL": "❌",
            "GOOD": "✅", "WARN": "⚠️ "}.get(status, "❓")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple exam scans")
    parser.add_argument("--template", required=True)
    parser.add_argument("--images",   required=True, nargs="+")
    parser.add_argument("--output",   default="output/comparison")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run verification on each
    from exam_verifier import verify_exam

    import logging
    logging.basicConfig(level=logging.WARNING)  # quiet for batch

    reports = []
    print("\n" + "="*80)
    print("  EXAM COMPARISON REPORT")
    print("="*80)

    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"  ⚠️  Skipping (not found): {img_path}")
            continue

        img_name = Path(img_path).stem
        print(f"\n  Processing: {img_name}...")

        try:
            report = verify_exam(
                image_path=img_path,
                template_path=args.template,
                output_dir=str(out_dir / img_name),
                debug=False
            )
            reports.append((img_name, report))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    # Print comparison table
    if not reports:
        print("No reports generated.")
        return

    print("\n" + "="*80)
    print("  COMPARISON SUMMARY")
    print("="*80)
    print(f"\n  {'Exam':<35}  {'Status':<8}  {'Overall':>8}  "
          f"{'Scan':>6}  {'Align':>6}  {'Mark':>6}  "
          f"{'Warnings':>8}")
    print(f"  {'─'*80}")

    for name, r in reports:
        scores  = r["scores"]
        n_warn  = len(r.get("warnings", []))
        status  = scores["overall_status"]
        icon    = _icon(status)
        conf    = scores["overall_confidence"]
        sq      = scores["scan_quality"]
        al      = scores["alignment"]
        mk      = scores["marking_clarity"]

        print(
            f"  {name[:35]:<35}  {icon} {status:<6}  "
            f"{conf:>7.1%}  {sq:>5.1%}  {al:>5.1%}  {mk:>5.1%}  "
            f"{n_warn:>8}"
        )

    # Per-question comparison
    if len(reports) > 1:
        print(f"\n\n  PER-QUESTION COMPARISON")
        print(f"  {'─'*80}")

        # Get question numbers from first report
        first_questions = sorted(
            reports[0][1]["questions"].keys(), key=int
        )

        # Header
        header = f"  {'Q':<4}"
        for name, _ in reports:
            short = name[:12]
            header += f"  {short:<14}"
        print(header)
        print(f"  {'─'*80}")

        for q_num in first_questions:
            row = f"  {q_num:<4}"
            for name, r in reports:
                qr = r["questions"].get(str(q_num), {})
                if not qr:
                    row += f"  {'─':<14}"
                    continue

                conf = qr.get("marking_confidence", 0)
                sel  = qr.get("selected_options", [])
                dbl  = qr.get("is_double_marked", False)
                emp  = qr.get("is_unmarked", False)

                if dbl:
                    flag = "⚠️ DBL"
                elif emp:
                    flag = "⬜ EMPTY"
                elif conf >= 0.80:
                    flag = f"✅{','.join(sel)}"
                else:
                    flag = f"~{','.join(sel)}"

                cell = f"{conf:.0%} {flag}"
                row += f"  {cell:<14}"
            print(row)

    # Save combined report
    combined = {
        "comparison": [
            {
                "name": name,
                "overall_status": r["scores"]["overall_status"],
                "overall_confidence": r["scores"]["overall_confidence"],
                "scan_quality": r["scores"]["scan_quality"],
                "alignment": r["scores"]["alignment"],
                "marking_clarity": r["scores"]["marking_clarity"],
                "warnings_count": len(r.get("warnings", []))
            }
            for name, r in reports
        ]
    }
    combined_path = out_dir / "comparison_summary.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n  📁 Comparison saved: {combined_path}")
    print()


if __name__ == "__main__":
    main()