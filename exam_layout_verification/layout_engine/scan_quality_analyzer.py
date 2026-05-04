"""
Scan Quality Analyzer

Answers the question: "Can we reliably read this scan?"

Checks:
  1. Blur level      → is image too blurry to OCR?
  2. Brightness      → too dark or too bright?
  3. Rotation angle  → is page tilted?
  4. Contrast        → enough difference between ink and paper?
  5. Noise level     → too much scanner noise?

Returns a ScanQualityReport with:
  - per-check scores (0.0 → 1.0)
  - overall scan quality score
  - human-readable warnings
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Thresholds ────────────────────────────────────────────────────────────────

# Laplacian variance: below this = blurry
BLUR_THRESHOLD_FAIL    = 30.0
BLUR_THRESHOLD_WARN    = 80.0
BLUR_THRESHOLD_GOOD    = 200.0

# Brightness (mean pixel 0-255)
BRIGHTNESS_TOO_DARK    = 60
BRIGHTNESS_TOO_BRIGHT  = 252
BRIGHTNESS_IDEAL_LOW   = 180
BRIGHTNESS_IDEAL_HIGH  = 250

# Rotation angle (degrees): above this = needs correction
ROTATION_WARN_DEG      = 1.0
ROTATION_FAIL_DEG      = 5.0

# Contrast (std dev of grayscale): below this = low contrast
CONTRAST_FAIL          = 10.0
CONTRAST_WARN          = 20.0

# Noise (high-freq energy ratio)
NOISE_FAIL             = 0.4
NOISE_WARN             = 0.25


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    score: float          # 0.0 = fail, 1.0 = perfect
    value: float          # raw measured value
    unit: str             # e.g. "px²", "°", ""
    status: str           # "GOOD" / "WARN" / "FAIL"
    message: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "value": round(self.value, 4),
            "unit": self.unit,
            "status": self.status,
            "message": self.message
        }


@dataclass
class ScanQualityReport:
    overall_score: float           # 0.0 → 1.0
    overall_status: str            # GOOD / WARN / FAIL
    can_process: bool              # True if we think we can proceed
    checks: List[CheckResult]
    warnings: List[str]
    rotation_deg: float            # detected rotation
    image_size: Tuple[int, int]    # (width, height)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "overall_status": self.overall_status,
            "can_process": self.can_process,
            "rotation_deg": round(self.rotation_deg, 3),
            "image_size": {"w": self.image_size[0], "h": self.image_size[1]},
            "warnings": self.warnings,
            "checks": [c.to_dict() for c in self.checks]
        }

    @property
    def blur_score(self) -> float:
        return next(
            (c.score for c in self.checks if c.name == "blur"), 0.0
        )

    @property
    def rotation_score(self) -> float:
        return next(
            (c.score for c in self.checks if c.name == "rotation"), 0.0
        )


# ── Main analyzer ─────────────────────────────────────────────────────────────

class ScanQualityAnalyzer:
    """
    Analyzes a scanned exam image for quality issues.
    Works entirely with OpenCV - no external dependencies.
    """

    def analyze(self, image: np.ndarray) -> ScanQualityReport:
        """
        Run all quality checks on the image.

        Args:
            image: BGR numpy array from cv2.imread

        Returns:
            ScanQualityReport
        """
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        checks: List[CheckResult] = []
        warnings: List[str] = []

        # Run all checks
        blur_check      = self._check_blur(gray)
        brightness_check = self._check_brightness(gray)
        contrast_check  = self._check_contrast(gray)
        rotation_check, rotation_deg = self._check_rotation(gray)
        noise_check     = self._check_noise(gray)

        checks = [
            blur_check,
            brightness_check,
            contrast_check,
            rotation_check,
            noise_check
        ]

        # Collect warnings
        for check in checks:
            if check.status == "FAIL":
                warnings.append(f"❌ {check.name.upper()}: {check.message}")
            elif check.status == "WARN":
                warnings.append(f"⚠️  {check.name.upper()}: {check.message}")

        # Overall score: weighted average
        weights = {
            "blur":       0.35,
            "brightness": 0.15,
            "contrast":   0.20,
            "rotation":   0.20,
            "noise":      0.10,
        }
        overall = sum(
            c.score * weights.get(c.name, 0.1)
            for c in checks
        )
        overall = round(min(1.0, max(0.0, overall)), 4)

        # Status
        fail_count = sum(1 for c in checks if c.status == "FAIL")
        warn_count = sum(1 for c in checks if c.status == "WARN")

        if fail_count >= 2 or blur_check.status == "FAIL":
            overall_status = "FAIL"
            can_process = False
        elif fail_count >= 1 or warn_count >= 3:
            overall_status = "WARN"
            can_process = True
        else:
            overall_status = "GOOD"
            can_process = True

        report = ScanQualityReport(
            overall_score=overall,
            overall_status=overall_status,
            can_process=can_process,
            checks=checks,
            warnings=warnings,
            rotation_deg=rotation_deg,
            image_size=(img_w, img_h)
        )

        logger.info(
            f"Scan quality: {overall_status} "
            f"(score={overall:.2f}, "
            f"blur={blur_check.value:.0f}, "
            f"rotation={rotation_deg:.2f}°)"
        )

        return report

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_blur(self, gray: np.ndarray) -> CheckResult:
        """
        Laplacian variance = sharpness measure.
        High = sharp, Low = blurry.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())

        if variance >= BLUR_THRESHOLD_GOOD:
            score = 1.0
            status = "GOOD"
            message = f"Sharp image (variance={variance:.0f})"
        elif variance >= BLUR_THRESHOLD_WARN:
            # Linear interpolation between WARN and GOOD
            t = (variance - BLUR_THRESHOLD_WARN) / (BLUR_THRESHOLD_GOOD - BLUR_THRESHOLD_WARN)
            score = 0.6 + 0.4 * t
            status = "WARN"
            message = f"Slightly blurry (variance={variance:.0f}) - may affect OCR"
        elif variance >= BLUR_THRESHOLD_FAIL:
            t = (variance - BLUR_THRESHOLD_FAIL) / (BLUR_THRESHOLD_WARN - BLUR_THRESHOLD_FAIL)
            score = 0.2 + 0.4 * t
            status = "WARN"
            message = f"Blurry image (variance={variance:.0f}) - OCR may fail"
        else:
            score = max(0.0, variance / BLUR_THRESHOLD_FAIL * 0.2)
            status = "FAIL"
            message = f"Very blurry (variance={variance:.0f}) - cannot reliably process"

        return CheckResult(
            name="blur", score=score, value=variance,
            unit="var", status=status, message=message
        )

    def _check_brightness(self, gray: np.ndarray) -> CheckResult:
        """Mean pixel brightness."""
        mean_brightness = float(gray.mean())

        if BRIGHTNESS_IDEAL_LOW <= mean_brightness <= BRIGHTNESS_IDEAL_HIGH:
            score = 1.0
            status = "GOOD"
            message = f"Good brightness (mean={mean_brightness:.0f})"
        elif mean_brightness < BRIGHTNESS_TOO_DARK:
            score = max(0.0, mean_brightness / BRIGHTNESS_TOO_DARK * 0.3)
            status = "FAIL"
            message = f"Too dark (mean={mean_brightness:.0f}) - increase scanner brightness"
        elif mean_brightness > BRIGHTNESS_TOO_BRIGHT:
            score = max(0.0, 1.0 - (mean_brightness - BRIGHTNESS_TOO_BRIGHT) / 25.0)
            status = "FAIL" if mean_brightness > 245 else "WARN"
            message = f"Too bright/overexposed (mean={mean_brightness:.0f})"
        else:
            # Between ideal and extremes: linear interpolation
            if mean_brightness < BRIGHTNESS_IDEAL_LOW:
                t = (mean_brightness - BRIGHTNESS_TOO_DARK) / (BRIGHTNESS_IDEAL_LOW - BRIGHTNESS_TOO_DARK)
            else:
                t = 1.0 - (mean_brightness - BRIGHTNESS_IDEAL_HIGH) / (BRIGHTNESS_TOO_BRIGHT - BRIGHTNESS_IDEAL_HIGH)
            score = 0.6 + 0.4 * t
            status = "WARN"
            message = f"Suboptimal brightness (mean={mean_brightness:.0f})"

        return CheckResult(
            name="brightness", score=score, value=mean_brightness,
            unit="", status=status, message=message
        )

    def _check_contrast(self, gray: np.ndarray) -> CheckResult:
        """Std deviation of pixel values = contrast measure."""
        std = float(gray.std())

        if std >= CONTRAST_WARN * 1.5:
            score = 1.0
            status = "GOOD"
            message = f"Good contrast (std={std:.1f})"
        elif std >= CONTRAST_WARN:
            t = (std - CONTRAST_WARN) / (CONTRAST_WARN * 0.5)
            score = 0.7 + 0.3 * t
            status = "GOOD"
            message = f"Acceptable contrast (std={std:.1f})"
        elif std >= CONTRAST_FAIL:
            t = (std - CONTRAST_FAIL) / (CONTRAST_WARN - CONTRAST_FAIL)
            score = 0.3 + 0.4 * t
            status = "WARN"
            message = f"Low contrast (std={std:.1f}) - text may be faint"
        else:
            score = max(0.0, std / CONTRAST_FAIL * 0.3)
            status = "FAIL"
            message = f"Very low contrast (std={std:.1f}) - cannot distinguish ink from paper"

        return CheckResult(
            name="contrast", score=score, value=std,
            unit="std", status=status, message=message
        )

    def _check_rotation(
        self, gray: np.ndarray
    ) -> Tuple[CheckResult, float]:
        """
        Estimate page rotation using Hough lines on edges.
        Returns (CheckResult, rotation_angle_degrees).
        """
        rotation_deg = self._estimate_rotation(gray)
        abs_rot = abs(rotation_deg)

        if abs_rot <= ROTATION_WARN_DEG:
            score = 1.0
            status = "GOOD"
            message = f"Straight scan (rotation={rotation_deg:+.2f}°)"
        elif abs_rot <= ROTATION_FAIL_DEG:
            t = 1.0 - (abs_rot - ROTATION_WARN_DEG) / (ROTATION_FAIL_DEG - ROTATION_WARN_DEG)
            score = 0.5 + 0.5 * t
            status = "WARN"
            message = f"Slightly rotated ({rotation_deg:+.2f}°) - alignment may be off"
        else:
            score = max(0.0, 0.5 - (abs_rot - ROTATION_FAIL_DEG) / 10.0)
            status = "FAIL"
            message = f"Significantly rotated ({rotation_deg:+.2f}°) - will cause misalignment"

        check = CheckResult(
            name="rotation", score=score, value=rotation_deg,
            unit="°", status=status, message=message
        )
        return check, rotation_deg

    def _check_noise(self, gray: np.ndarray) -> CheckResult:
        """
        Estimate noise by comparing image to Gaussian-blurred version.
        High difference = noisy.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred)
        noise_ratio = float(diff.mean()) / 255.0

        if noise_ratio <= NOISE_WARN * 0.5:
            score = 1.0
            status = "GOOD"
            message = f"Clean scan (noise={noise_ratio:.3f})"
        elif noise_ratio <= NOISE_WARN:
            t = 1.0 - (noise_ratio - NOISE_WARN * 0.5) / (NOISE_WARN * 0.5)
            score = 0.7 + 0.3 * t
            status = "GOOD"
            message = f"Low noise (noise={noise_ratio:.3f})"
        elif noise_ratio <= NOISE_FAIL:
            t = 1.0 - (noise_ratio - NOISE_WARN) / (NOISE_FAIL - NOISE_WARN)
            score = 0.3 + 0.4 * t
            status = "WARN"
            message = f"Moderate noise (noise={noise_ratio:.3f}) - may affect detection"
        else:
            score = max(0.0, 0.3 - (noise_ratio - NOISE_FAIL) * 0.5)
            status = "FAIL"
            message = f"Heavy noise (noise={noise_ratio:.3f}) - scan quality is poor"

        return CheckResult(
            name="noise", score=score, value=noise_ratio,
            unit="", status=status, message=message
        )

    def _estimate_rotation(self, gray: np.ndarray) -> float:
        """
        Estimate rotation angle using:
        1. Hough lines on Canny edges (primary)
        2. Minarea rect of text contours (fallback)
        """
        # Method 1: Hough lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            angles = []
            for line in lines[:30]:  # Use top 30 lines
                rho, theta = line[0]
                # Convert to degrees from vertical
                angle_deg = math.degrees(theta) - 90
                # Filter: keep near-horizontal or near-vertical lines
                if abs(angle_deg) < 45:
                    angles.append(angle_deg)

            if angles:
                # Use median to ignore outliers
                return float(np.median(angles))

        # Method 2: Minarea rect fallback
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
            rect = cv2.minAreaRect(all_pts)
            angle = rect[2]
            if angle < -45:
                angle += 90
            return float(angle)

        return 0.0