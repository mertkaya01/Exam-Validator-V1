# 2200004016
# Mert KAYA
# Software Engineering


from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# Handlers are attached only when running as __main__ (see bottom of file).
# Library consumers manage their own logging configuration.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions

class ExamParserError(Exception):
    """Base exception for all errors raised by this module."""

class JSONLoadError(ExamParserError):
    """Raised when the exam JSON file cannot be read or parsed."""

class ImageLoadError(ExamParserError):
    """Raised when an image file cannot be opened by OpenCV."""

class InvalidPageDataError(ExamParserError):
    """Raised when a page entry is missing required structural keys."""


# ---------------------------------------------------------------------------
# Colour palette  (OpenCV BGR order)

class Color:
    """BGR colour constants used for annotating exam regions.

    All values are ``(Blue, Green, Red)`` tuples compatible with OpenCV's
    ``cv2.rectangle`` / ``cv2.circle`` calls.
    """

    ANCHOR: Tuple[int, int, int] = (0, 0, 220)         # Red   – corner anchors
    STUDENT_REGION: Tuple[int, int, int] = (180, 0, 180)   # Purple – student-number region
    STUDENT_BOX: Tuple[int, int, int] = (220, 50, 220)     # Light purple – individual digit boxes
    FULL_NAME: Tuple[int, int, int] = (180, 180, 0)        # Teal   – student full-name region
    QUESTION_BOX: Tuple[int, int, int] = (0, 180, 0)       # Green  – question bounding boxes
    OPTION: Tuple[int, int, int] = (220, 100, 0)           # Blue   – answer-option checkboxes
    SOLUTION_AREA: Tuple[int, int, int] = (0, 140, 255)    # Orange – open-ended solution areas
    MATCHING_BOX: Tuple[int, int, int] = (0, 220, 220)     # Yellow – matching answer boxes
    FILL_BLANK: Tuple[int, int, int] = (100, 255, 100)     # Light green – fill-in-the-blank slots
    LABEL_BG: Tuple[int, int, int] = (30, 30, 30)          # Dark grey – label background
    LABEL_TEXT: Tuple[int, int, int] = (255, 255, 255)     # White  – label text

# ---------------------------------------------------------------------------
# Geometric transform context

@dataclass
class TransformContext:
    """Context for geometric transformations."""

    sx: float = 1.0
    sy: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    homography: Optional[np.ndarray] = field(default=None, repr=False)

    def apply_point(self, x: float, y: float) -> Tuple[int, int]:
        """Map a JSON (x, y) point to image pixel coordinates."""
        if self.homography is not None:
            src = np.array([[[x, y]]], dtype=np.float64)
            dst = cv2.perspectiveTransform(src, self.homography)
            return int(round(dst[0, 0, 0])), int(round(dst[0, 0, 1]))
        return (
            int(round(x * self.sx + self.offset_x)),
            int(round(y * self.sy + self.offset_y)),
        )

    def apply_rect_vertices(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> np.ndarray:
        """Dikdörtgenin 4 köşesini de transform ederek bir poligon oluşturur."""
        # Köşeleri saat yönünde tanımlıyoruz
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)

        if self.homography is not None:
            # Homografi varsa 4 noktayı da perspektif olarak döndürür
            dst = cv2.perspectiveTransform(pts, self.homography)
            return dst.astype(np.int32)
        
        # Homografi yoksa (sadece offset varsa) her noktaya sx, sy ve offset uygula
        res = []
        for p in [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]:
            px, py = self.apply_point(p[0], p[1])
            res.append([px, py])
        return np.array(res, dtype=np.int32).reshape(-1, 1, 2)


# ---------------------------------------------------------------------------
# Anchor detection helpers

def _detect_anchor_center(
    gray: np.ndarray,
    expected_px: int,
    expected_py: int,
    search_radius: int,
) -> Optional[Tuple[float, float]]:
    img_h, img_w = gray.shape
    # Search for anchor center in a square around the expected position

    # Clamp ROI to image boundaries
    roi_x1 = max(0, expected_px - search_radius)
    roi_y1 = max(0, expected_py - search_radius)
    roi_x2 = min(img_w, expected_px + search_radius)
    roi_y2 = min(img_h, expected_py + search_radius)

    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        logger.debug(
            "Anchor search ROI is empty at expected=(%d, %d).",
            expected_px,
            expected_py,
        )
        return None

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # --- Strategy 1: Hough Circle Transform ---
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=60,
        param2=18,
        minRadius=4,
        maxRadius=max(5, search_radius // 2),
    )
    if circles is not None:
        # Select the largest (most prominent) circle
        best = max(circles[0], key=lambda c: c[2])
        return float(roi_x1 + best[0]), float(roi_y1 + best[1])

    # --- Strategy 2: Largest-contour centroid fallback ---
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            return float(roi_x1 + cx), float(roi_y1 + cy)

    logger.debug(
        "No anchor detected at expected=(%d, %d).", expected_px, expected_py
    )
    return None


def compute_transform(
    image: np.ndarray,
    anchors: Dict[str, dict],
    scale_x: float,
    scale_y: float,
) -> TransformContext:
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    # Search radius: 5 % of the shorter page dimension, clamped to [40, 200]
    search_radius = int(np.clip(min(img_w, img_h) * 0.05, 40, 200))

    src_points: List[List[float]] = []   # JSON reference coordinates
    dst_points: List[List[float]] = []   # Detected pixel coordinates

    for corner_name, anchor_data in anchors.items():
        center = anchor_data.get("center", {})
        ref_x = float(center.get("x", 0))
        ref_y = float(center.get("y", 0))

        exp_px = int(round(ref_x * scale_x))
        exp_py = int(round(ref_y * scale_y))

        detected = _detect_anchor_center(gray, exp_px, exp_py, search_radius)

        if detected:
            src_points.append([ref_x, ref_y])
            dst_points.append(list(detected))
            logger.debug(
                "Anchor '%s': JSON=(%.1f, %.1f) | expected=(%d, %d) | "
                "detected=(%.1f, %.1f).",
                corner_name,
                ref_x,
                ref_y,
                exp_px,
                exp_py,
                *detected,
            )
        else:
            logger.debug(
                "Anchor '%s': NOT detected (expected pixel=(%d, %d)).",
                corner_name,
                exp_px,
                exp_py,
            )

    n_detected = len(src_points)
    logger.info("Anchor detection: %d / %d corners found.", n_detected, len(anchors))

    ctx = TransformContext(sx=scale_x, sy=scale_y)

    if n_detected >= 4:
        src_np = np.array(src_points, dtype=np.float64)
        dst_np = np.array(dst_points, dtype=np.float64)
        homography, _ = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 3.0)
        if homography is not None:
            ctx.homography = homography
            logger.info(
                "Transform mode: Perspective Homography (RANSAC, %d points).",
                n_detected,
            )
            return ctx
        logger.warning(
            "Homography computation failed; falling back to affine offset."
        )

    if n_detected >= 2:
        offsets_x = [
            dst_points[i][0] - src_points[i][0] * scale_x
            for i in range(n_detected)
        ]
        offsets_y = [
            dst_points[i][1] - src_points[i][1] * scale_y
            for i in range(n_detected)
        ]
        ctx.offset_x = float(np.median(offsets_x))
        ctx.offset_y = float(np.median(offsets_y))
        logger.info(
            "Transform mode: Affine Offset  offset_x=%.2f  offset_y=%.2f "
            "(%d-point median).",
            ctx.offset_x,
            ctx.offset_y,
            n_detected,
        )
        return ctx

    logger.warning(
        "Insufficient anchors detected (%d/%d). Using pure scale — "
        "scan-margin drift will NOT be corrected.",
        n_detected,
        len(anchors),
    )
    return ctx


# ---------------------------------------------------------------------------
# Drawing primitives

def draw_rect(
    image: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    ctx: TransformContext,
    color: Tuple[int, int, int],
    thickness: int = 2,
    fill_alpha: float = 0.0,
) -> Tuple[int, int]:
    
    # Yeni fonksiyonumuzla 4 köşeyi alıyoruz
    pts = ctx.apply_rect_vertices(x, y, w, h)

    if fill_alpha > 0.0:
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, image, 1.0 - fill_alpha, 0, image)

    # Düz kutu yerine bükülebilir poligon çiziyoruz
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    
    # Label için sol üst köşeyi döndür (yine eskisi gibi çalışsın diye)
    return tuple(pts[0][0])


def draw_bullseye(
    image: np.ndarray,
    cx: float,
    cy: float,
    diameter: float,
    ctx: TransformContext,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> Tuple[int, int]:
    
    px, py = ctx.apply_point(cx, cy)
    avg_scale = (ctx.sx + ctx.sy) / 2.0
    radius = max(3, int(round((diameter / 2.0) * avg_scale)))

    cv2.circle(image, (px, py), radius, color, thickness)
    cv2.circle(image, (px, py), max(2, radius // 3), color, cv2.FILLED)
    return px, py


def draw_label(
    image: np.ndarray,
    text: str,
    px: int,
    py: int,
    text_color: Tuple[int, int, int] = Color.LABEL_TEXT,
    bg_color: Tuple[int, int, int] = Color.LABEL_BG,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    baseline_y = py - 4
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    cv2.rectangle(
        image,
        (px - 1, baseline_y - text_h - baseline),
        (px + text_w + 1, baseline_y + baseline),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(
        image, text, (px, baseline_y), font, font_scale, text_color, thickness,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main class

class ExamLayoutParser:
    

    def __init__(self, json_path: str, output_dir: str = ".") -> None:
        self._json_path = json_path
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._exam_data: dict = self._load_json(json_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, image_paths: Optional[List[str]] = None) -> List[Path]:
        
        pages = self._exam_data.get("pages", [])
        if not pages:
            raise ExamParserError(
                f"No 'pages' array found in exam JSON: {self._json_path}"
            )

        image_paths = image_paths or []
        output_paths: List[Path] = []

        logger.info(
            "Starting annotation — examId='%s', total pages=%d.",
            self._exam_data.get("examId", "unknown"),
            len(pages),
        )

        for page_index, page_data in enumerate(pages):
            img_path = (
                image_paths[page_index] if page_index < len(image_paths) else None
            )
            logger.info(
                "Processing page %d / %d ...", page_index + 1, len(pages)
            )
            try:
                out_path = self._process_page(page_data, img_path, page_index)
                output_paths.append(out_path)
            except (InvalidPageDataError, ImageLoadError) as exc:
                logger.error(
                    "Skipping page %d due to error: %s", page_index + 1, exc
                )

        logger.info(
            "Annotation complete. %d / %d pages saved to '%s'.",
            len(output_paths),
            len(pages),
            self._output_dir,
        )
        return output_paths

    # ------------------------------------------------------------------
    # JSON loading

    @staticmethod
    def _load_json(path: str) -> dict:
        
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError as exc:
            raise JSONLoadError(f"Exam JSON not found: {path}") from exc
        except PermissionError as exc:
            raise JSONLoadError(
                f"Permission denied when reading JSON: {path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise JSONLoadError(
                f"Invalid JSON in file '{path}': {exc}"
            ) from exc

        logger.info(
            "JSON loaded — examId='%s', pages=%s.",
            data.get("examId", "N/A"),
            data.get("totalPages", "N/A"),
        )
        return data

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        
        image = cv2.imread(path)
        if image is None:
            raise ImageLoadError(
                f"OpenCV could not read image (unsupported format or missing "
                f"file): {path}"
            )
        h, w = image.shape[:2]
        logger.info("Image loaded: '%s'  (%d × %d px).", path, w, h)
        return image

    # ------------------------------------------------------------------
    # Per-page orchestration

    def _process_page(
        self,
        page_data: dict,
        image_path: Optional[str],
        page_index: int,
    ) -> Path:
        
        ref_w = float(page_data.get("pageWidth", 756))
        ref_h = float(page_data.get("pageHeight", 1086))

        # Load the real scan or create a blank white canvas for dry-runs
        if image_path and os.path.exists(image_path):
            image = self._load_image(image_path)
        else:
            if image_path:
                logger.warning(
                    "Image not found at '%s'. Using blank canvas for page %d.",
                    image_path,
                    page_index + 1,
                )
            else:
                logger.info(
                    "No image supplied for page %d. Using blank canvas.",
                    page_index + 1,
                )
            image = np.full(
                (int(ref_h), int(ref_w), 3), 255, dtype=np.uint8
            )

        img_h, img_w = image.shape[:2]
        scale_x = img_w / ref_w
        scale_y = img_h / ref_h

        # Compute anchor-based geometric transform
        anchors: Dict[str, dict] = page_data.get("anchors", {})
        if anchors:
            ctx = compute_transform(image, anchors, scale_x, scale_y)
        else:
            logger.info(
                "No anchors on page %d — using pure scale transform.",
                page_index + 1,
            )
            ctx = TransformContext(sx=scale_x, sy=scale_y)

        # Draw all annotated regions
        if anchors:
            self._draw_anchors(image, anchors, ctx)

        self._draw_identity_regions(image, page_data, ctx)

        questions: Dict[str, dict] = page_data.get("questions", {})
        if questions:
            self._draw_questions(image, questions, ctx)

        # Persist annotated image
        page_id = page_data.get("pageId", f"page_{page_index + 1}")
        output_filename = f"annotated_{page_id}.jpg"
        output_path = self._output_dir / output_filename
        success = cv2.imwrite(
            str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not success:
            raise ExamParserError(
                f"OpenCV failed to write image to: {output_path}"
            )

        logger.info("Saved annotated image: '%s'.", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Region drawers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_anchors(
        image: np.ndarray,
        anchors: Dict[str, dict],
        ctx: TransformContext,
    ) -> None:
        
        for corner_name, anchor_data in anchors.items():
            center = anchor_data.get("center", {})
            diameter = float(anchor_data.get("diameter", 16))
            cx = float(center.get("x", 0))
            cy = float(center.get("y", 0))

            px, py = draw_bullseye(image, cx, cy, diameter, ctx, Color.ANCHOR)
            label_y = py - max(3, int((diameter / 2.0) * (ctx.sx + ctx.sy) / 2.0))
            draw_label(image, f"ANC-{corner_name}", px, label_y)

    @staticmethod
    def _draw_identity_regions(
        image: np.ndarray,
        page_data: dict,
        ctx: TransformContext,
    ) -> None:
        
        number_region: Optional[dict] = page_data.get("studentNumberRegion")

        # --- Student number region (outer border) ---
        if number_region:
            px, py = draw_rect(
                image,
                number_region["x"],
                number_region["y"],
                number_region["w"],
                number_region["h"],
                ctx,
                Color.STUDENT_REGION,
                thickness=2,
                fill_alpha=0.05,
            )
            draw_label(image, "STUDENT_REGION", px, py)

        # --- Individual digit boxes ---
        for digit_box in page_data.get("studentNumberBoxes", []):
            draw_rect(
                image,
                digit_box["x"],
                digit_box["y"],
                digit_box["w"],
                digit_box["h"],
                ctx,
                Color.STUDENT_BOX,
                thickness=1,
            )

        # --- Full-name region ---
        name_region: Optional[dict] = (
            page_data.get("fullNameRegion")
            or page_data.get("studentNameRegion")
            or page_data.get("nameRegion")
        )
        is_inferred = False

        if name_region is None and number_region:
            # Infer: the horizontal strip to the left of the student-number region
            inferred_width = number_region["x"] - 20.0  # 10 px left margin + 10 gap
            if inferred_width > 40:
                name_region = {
                    "x": 10.0,
                    "y": number_region["y"],
                    "w": inferred_width,
                    "h": number_region["h"],
                }
                is_inferred = True
                logger.info(
                    "fullNameRegion not found in JSON — inferred automatically: "
                    "x=10, y=%.1f, w=%.1f, h=%.1f.",
                    number_region["y"],
                    inferred_width,
                    number_region["h"],
                )

        if name_region:
            label = "FULL_NAME_REGION" + (" [inferred]" if is_inferred else "")
            px, py = draw_rect(
                image,
                name_region["x"],
                name_region["y"],
                name_region["w"],
                name_region["h"],
                ctx,
                Color.FULL_NAME,
                thickness=2,
                fill_alpha=0.06,
            )
            draw_label(image, label, px, py)

    @staticmethod
    def _draw_questions(
        image: np.ndarray,
        questions: Dict[str, dict],
        ctx: TransformContext,
    ) -> None:
        
        for q_number, q_data in questions.items():
            if not isinstance(q_data, dict):
                logger.warning(
                    "Question '%s' has unexpected data type %s — skipping.",
                    q_number,
                    type(q_data).__name__,
                )
                continue

            q_type: str = q_data.get("type", "unknown")
            bounding_box: dict = q_data.get("boundingBox", {})

            # Always draw the outer bounding box if present
            if bounding_box:
                try:
                    px, py = draw_rect(
                        image,
                        bounding_box["x"],
                        bounding_box["y"],
                        bounding_box["w"],
                        bounding_box["h"],
                        ctx,
                        Color.QUESTION_BOX,
                        thickness=3,
                        fill_alpha=0.04,
                    )
                    draw_label(image, f"Q{q_number} [{q_type}]", px, py)
                except KeyError as exc:
                    logger.warning(
                        "Question '%s' boundingBox is missing key %s.",
                        q_number,
                        exc,
                    )

            # Dispatch to type-specific renderers
            if q_type in ("multiple_choice", "multi_select"):
                ExamLayoutParser._draw_options(image, q_number, q_data, ctx)

            elif q_type == "open_ended":
                ExamLayoutParser._draw_solution_area(image, q_number, q_data, ctx)

            elif q_type == "matching":
                ExamLayoutParser._draw_matching_boxes(image, q_number, q_data, ctx)

            elif q_type == "fill_blanks":
                ExamLayoutParser._draw_fill_blanks(image, q_number, q_data, ctx)

            elif q_type != "unknown":
                logger.debug(
                    "Question '%s' has unrecognised type '%s' — "
                    "only bounding box drawn.",
                    q_number,
                    q_type,
                )

    # ------------------------------------------------------------------
    # Question-type sub-renderers

    @staticmethod
    def _draw_options(
        image: np.ndarray,
        q_number: str,
        q_data: dict,
        ctx: TransformContext,
    ) -> None:
        
        for option_key, option_box in q_data.get("options", {}).items():
            try:
                opx, opy = draw_rect(
                    image,
                    option_box["x"],
                    option_box["y"],
                    option_box["w"],
                    option_box["h"],
                    ctx,
                    Color.OPTION,
                    thickness=1,
                )
                draw_label(image, option_key, opx, opy, font_scale=0.35)
            except KeyError as exc:
                logger.warning(
                    "Q%s option '%s' missing key %s — skipped.",
                    q_number,
                    option_key,
                    exc,
                )

    @staticmethod
    def _draw_solution_area(
        image: np.ndarray,
        q_number: str,
        q_data: dict,
        ctx: TransformContext,
    ) -> None:
        
        solution_area: Optional[dict] = q_data.get("solutionArea")
        if not solution_area:
            return
        try:
            spx, spy = draw_rect(
                image,
                solution_area["x"],
                solution_area["y"],
                solution_area["w"],
                solution_area["h"],
                ctx,
                Color.SOLUTION_AREA,
                thickness=2,
                fill_alpha=0.06,
            )
            draw_label(image, f"Q{q_number} SOLUTION", spx, spy)
        except KeyError as exc:
            logger.warning(
                "Q%s solutionArea missing key %s — skipped.", q_number, exc
            )

    @staticmethod
    def _draw_matching_boxes(
        image: np.ndarray,
        q_number: str,
        q_data: dict,
        ctx: TransformContext,
    ) -> None:
        
        answer_section: Optional[dict] = q_data.get("answerSection")
        if answer_section:
            try:
                draw_rect(
                    image,
                    answer_section["x"],
                    answer_section["y"],
                    answer_section["w"],
                    answer_section["h"],
                    ctx,
                    Color.MATCHING_BOX,
                    thickness=1,
                    fill_alpha=0.04,
                )
            except KeyError as exc:
                logger.warning(
                    "Q%s answerSection missing key %s.", q_number, exc
                )

        for slot_number, slot_box in q_data.get("answerBoxes", {}).items():
            try:
                bpx, bpy = draw_rect(
                    image,
                    slot_box["x"],
                    slot_box["y"],
                    slot_box["w"],
                    slot_box["h"],
                    ctx,
                    Color.MATCHING_BOX,
                    thickness=2,
                )
                draw_label(image, f"M{slot_number}", bpx, bpy, font_scale=0.38)
            except KeyError as exc:
                logger.warning(
                    "Q%s answerBox '%s' missing key %s — skipped.",
                    q_number,
                    slot_number,
                    exc,
                )

    @staticmethod
    def _draw_fill_blanks(
        image: np.ndarray,
        q_number: str,
        q_data: dict,
        ctx: TransformContext,
    ) -> None:
        
        answer_section: Optional[dict] = q_data.get("answerSection")
        if answer_section:
            try:
                draw_rect(
                    image,
                    answer_section["x"],
                    answer_section["y"],
                    answer_section["w"],
                    answer_section["h"],
                    ctx,
                    Color.FILL_BLANK,
                    thickness=1,
                    fill_alpha=0.04,
                )
            except KeyError as exc:
                logger.warning(
                    "Q%s fill-blanks answerSection missing key %s.", q_number, exc
                )

        for blank_id, blank_box in q_data.get("fillBlanks", {}).items():
            try:
                fpx, fpy = draw_rect(
                    image,
                    blank_box["x"],
                    blank_box["y"],
                    blank_box["w"],
                    blank_box["h"],
                    ctx,
                    Color.FILL_BLANK,
                    thickness=2,
                )
                draw_label(image, f"B{blank_id}", fpx, fpy, font_scale=0.38)
            except KeyError as exc:
                logger.warning(
                    "Q%s fillBlank '%s' missing key %s — skipped.",
                    q_number,
                    blank_id,
                    exc,
                )

        for box_id, answer_box in q_data.get("answerBoxes", {}).items():
            try:
                draw_rect(
                    image,
                    answer_box["x"],
                    answer_box["y"],
                    answer_box["w"],
                    answer_box["h"],
                    ctx,
                    Color.FILL_BLANK,
                    thickness=2,
                )
            except KeyError as exc:
                logger.warning(
                    "Q%s answerBox '%s' missing key %s — skipped.",
                    q_number,
                    box_id,
                    exc,
                )


# ---------------------------------------------------------------------------
# CLI

def _build_arg_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(
        prog="exam_layout_parser",
        description=(
            "Annotate exam page images using a structured JSON layout file.\n"
            "Each recognised region (anchors, student fields, questions, "
            "options, solution areas) is drawn with a distinct colour."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--json", "-j",
        required=True,
        metavar="EXAM_JSON",
        help="Path to the exam JSON file (e.g. exam.json).",
    )
    parser.add_argument(
        "--images", "-i",
        nargs="*",
        metavar="IMAGE",
        default=[],
        help=(
            "Ordered list of page image files (one per page).\n"
            "Example: --images page1.jpg page2.jpg"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        metavar="DIR",
        help="Directory for annotated images (default: ./output).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> None:
    """Entry point for CLI execution."""
    args = _build_arg_parser().parse_args()

    # Configure logging only when running as a script
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        parser = ExamLayoutParser(json_path=args.json, output_dir=args.output)
        parser.process(image_paths=args.images)
    except ExamParserError as exc:
        logger.error("Fatal error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()