"""
Creates test images from your existing scans:
  1. Rotated versions (simulating tilted paper)
  2. Simulated student marks on blank exam

Usage:
    python create_test_rotated.py --image examples/ICE101-midterm-2026.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image around center, keeping same canvas size."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255)  # white background
    )
    return rotated


def add_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Simulate motion blur / poor focus."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def add_noise(image: np.ndarray, intensity: int = 30) -> np.ndarray:
    """Add scanner noise."""
    noise = np.random.randint(
        -intensity, intensity,
        image.shape, dtype=np.int16
    )
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def darken(image: np.ndarray, factor: float = 0.7) -> np.ndarray:
    """Simulate dark scan."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def draw_circle_mark(
    image: np.ndarray,
    cx: int, cy: int,
    radius: int = 12,
    filled: bool = True
) -> None:
    """Draw a student mark (filled circle) on image."""
    if filled:
        cv2.circle(image, (cx, cy), radius, (20, 20, 20), -1)
        # Add slight imperfection (real marks aren't perfect)
        cv2.circle(image, (cx, cy), radius - 2, (40, 40, 40), 1)
    else:
        # Partial mark (ambiguous)
        cv2.ellipse(
            image, (cx, cy), (radius, radius),
            0, 0, 200,  # 200 degrees = partial
            (20, 20, 20), 2
        )


def simulate_student_marks(
    image: np.ndarray,
    template: dict,
    scale: float,
    answers: dict  # {question_num: "A"} or {1: "C", 2: "B"}
) -> np.ndarray:
    """
    Draw student marks on a blank exam image.
    
    Args:
        image: blank exam scan
        template: loaded JSON template
        scale: image_width / template_width
        answers: {question_number: selected_option_label}
    
    Returns:
        Image with marks drawn
    """
    marked = image.copy()
    page_data = template["pages"][0]
    questions = page_data.get("questions", {})

    for q_num_str, q_data in questions.items():
        q_num = int(q_num_str)
        q_type = q_data.get("type")
        answer = answers.get(q_num)

        if answer is None:
            continue

        if q_type in ("multiple_choice", "multi_select"):
            options = q_data.get("options", {})

            # Handle single answer or list of answers
            answer_list = [answer] if isinstance(answer, str) else answer

            for label in answer_list:
                if label in options:
                    opt = options[label]
                    cx = int((opt["x"] + opt["w"] / 2) * scale)
                    cy = int((opt["y"] + opt["h"] / 2) * scale)
                    radius = max(8, int(opt["w"] / 2 * scale * 0.7))
                    draw_circle_mark(marked, cx, cy, radius, filled=True)

    return marked


def main():
    parser = argparse.ArgumentParser(description="Create test images")
    parser.add_argument("--image",    required=True, help="Source scan PNG")
    parser.add_argument("--template", help="Template JSON (for marked version)")
    parser.add_argument("--output",   default="examples/test_variants",
                        help="Output directory")
    args = parser.parse_args()

    src = Path(args.image)
    if not src.exists():
        print(f"❌ Image not found: {src}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(src))
    if image is None:
        print(f"❌ Cannot load: {src}")
        sys.exit(1)

    stem = src.stem
    print(f"\n📸 Source: {src}  ({image.shape[1]}×{image.shape[0]}px)")
    print(f"📁 Output: {out_dir}/\n")

    # ── Generate variants ─────────────────────────────────────────────────────
    variants = {
        f"{stem}_clean":         image,
        f"{stem}_rot_1deg":      rotate_image(image, 1.0),
        f"{stem}_rot_2deg":      rotate_image(image, 2.0),
        f"{stem}_rot_5deg":      rotate_image(image, 5.0),
        f"{stem}_rot_minus2deg": rotate_image(image, -2.0),
        f"{stem}_blurry_light":  add_blur(image, 3),
        f"{stem}_blurry_heavy":  add_blur(image, 9),
        f"{stem}_noisy":         add_noise(image, 25),
        f"{stem}_dark":          darken(image, 0.6),
        f"{stem}_rot2_blurry":   add_blur(rotate_image(image, 2.0), 5),
    }

    # Add student-marked version if template provided
    if args.template:
        import json
        with open(args.template, "r") as f:
            template = json.load(f)

        page_w = template["pages"][0]["pageWidth"]
        scale  = image.shape[1] / page_w

        # Sample answers for marking
        questions = template["pages"][0].get("questions", {})
        sample_answers = {}
        option_cycle = ["A", "B", "C", "D"]

        for i, q_num_str in enumerate(sorted(questions.keys(), key=int)):
            q_num  = int(q_num_str)
            q_data = questions[q_num_str]
            q_type = q_data.get("type")

            if q_type == "multiple_choice":
                opts = list(q_data.get("options", {}).keys())
                if opts:
                    sample_answers[q_num] = opts[i % len(opts)]
            elif q_type == "multi_select":
                opts = list(q_data.get("options", {}).keys())
                if len(opts) >= 2:
                    sample_answers[q_num] = [opts[0], opts[2]]

        marked = simulate_student_marks(image, template, scale, sample_answers)
        variants[f"{stem}_student_marked"] = marked

        # Also rotated + marked
        variants[f"{stem}_marked_rot2deg"] = rotate_image(marked, 2.0)

        print(f"  Student answers simulated: {sample_answers}")

    # Save all variants
    for name, img in variants.items():
        path = out_dir / f"{name}.png"
        cv2.imwrite(str(path), img)
        print(f"  ✓ {name}.png")

    print(f"\n✅ Generated {len(variants)} test images")
    print("\nNow run verification on them:")
    print(f"  python exam_verifier.py --image {out_dir}/{stem}_rot_2deg.png --template <your_template.json>")
    print(f"  python exam_verifier.py --image {out_dir}/{stem}_student_marked.png --template <your_template.json>")


if __name__ == "__main__":
    main()