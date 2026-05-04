# 1. Generate test variants from your blank exam
python create_test_rotated.py \
  --image examples/ICE101-midterm-2026.png \
  --template templates/ICE101_midterm.json \
  --output examples/test_variants

# 2. Test the rotated ones
python exam_verifier.py \
  --image examples/test_variants/ICE101-midterm-2026_rot_2deg.png \
  --template templates/ICE101_midterm.json \
  --output output/rotated_test

# 3. Test with simulated student marks
python exam_verifier.py \
  --image examples/test_variants/ICE101-midterm-2026_student_marked.png \
  --template templates/ICE101_midterm.json \
  --output output/marked_test

# 4. Compare all variants side by side
python compare_exams.py \
  --template templates/ICE101_midterm.json \
  --images \
    examples/ICE101-midterm-2026.png \
    examples/test_variants/ICE101-midterm-2026_rot_2deg.png \
    examples/test_variants/ICE101-midterm-2026_student_marked.png \
    examples/test_variants/ICE101-midterm-2026_blurry_heavy.png \
    examples/test_variants/ICE101-midterm-2026_dark.png \
  --output output/comparison

This text file is included in the repository for your convenience. It contains a list of words and their frequencies. 
We are doing starting from zero to last piece.

1. PDF to PNG
python pdf_to_png.py --pdf samples/exam_paper.pdf --output examples --verbose

2. PNG to evaluation with JSON
python examlayoutparser.py --json exam_json.json --images page1.png --output ./outputs

# Test with rotated images
python .\create_test_rotated.py --image examples/ICE101-midterm-2026.png

# Test with run_validation.py for confidence points
python run_validation.py --image examples/ICE101-midterm-2026.png --template templates/ICE101_midterm.json --debug

3. Test the with exam_verifier = testing with blank sheet to filled sheet
python exam_verifier.py \
  --image examples/test_variants/ICE101-midterm-2026_rot_2deg.png \
  --template templates/ICE101_midterm.json \
  --output output/rotated_test

4. Compare all variants side by side
python compare_exams.py \
  --template templates/ICE101_midterm.json \
  --images \
    examples/ICE101-midterm-2026.png \
    examples/test_variants/ICE101-midterm-2026_rot_2deg.png \
    examples/test_variants/ICE101-midterm-2026_student_marked.png \
    examples/test_variants/ICE101-midterm-2026_blurry_heavy.png \
    examples/test_variants/ICE101-midterm-2026_dark.png \
  --output output/comparison