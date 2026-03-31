import cv2
import numpy as np
import torch
import time
from torch import nn
from pathlib import Path
from collections import deque

GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9

CELL_MARGIN_RATIO = 0.16
MIN_COMPONENT_AREA_RATIO = 0.015
MIN_COMPONENT_FILL_RATIO = 0.12
MIN_PREDICTION_CONFIDENCE = 0.58
MIN_CONFIDENCE_MARGIN = 0.06
CORNER_SMOOTHING_WINDOW = 50
CORNER_MAX_DRIFT = 8.0
DIGIT_SMOOTHING_WINDOW = 5
MIN_EMPTY_PIXEL_RATIO = 0.03
TEMPORAL_DECAY = 0.72
MODEL_IMAGE_SIZE = 28

MIN_CLUES_TO_SOLVE = 17
EMPTY_CELL_PROBABILITY_THRESHOLD = 0.52
LOW_DIGIT_PROBABILITY_THRESHOLD = 0.34
ABS_DIGIT_CANDIDATE_PROB = 0.03
RELATIVE_DIGIT_CANDIDATE_RATIO = 0.18
MAX_DIGIT_CANDIDATES = 5
SOLUTION_PERSISTENCE_FRAMES = 60
MIN_CORNER_MAX_DRIFT = 2.0
MAX_CORNER_MAX_DRIFT = 30.0
CORNER_DRIFT_STEP = 0.5
MIN_PERSISTENCE_FRAMES = 10
MAX_PERSISTENCE_FRAMES = 300
PERSISTENCE_STEP_FRAMES = 10

MODEL_ONNX_PATH = Path(__file__).with_name("digit_cnn.pth")

BLUR_KERNEL_SIZE = (7, 7)
THRESH_BLOCK_SIZE = 11
THRESH_C = 2
GRID_OPEN_KERNEL = np.ones((3, 3), dtype=np.uint8)
CELL_CLEAN_KERNEL = np.ones((2, 2), dtype=np.uint8)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

WARP_DESTINATION = np.array(
    [[0, 0], [GRID_SIZE - 1, 0], [GRID_SIZE - 1, GRID_SIZE - 1], [0, GRID_SIZE - 1]],
    dtype="float32",
)

print("=" * 50)
print("Initializing GPU Support...")
print("=" * 50)
print(f"OpenCV Version: {cv2.__version__}")

gpu_available = False
try:
    cv2.ocl.setUseOpenCL(True)
    gpu_available = bool(cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL())
    print(f"OpenCL Available: {cv2.ocl.haveOpenCL()}")
    print(f"OpenCL Enabled: {cv2.ocl.useOpenCL()}")
    if gpu_available:
        print("✓ GPU acceleration ENABLED via OpenCL")
    else:
        print("✗ OpenCL runtime not active, using CPU")
except Exception as error:
    print(f"✗ GPU Access Error: {error}")

print("=" * 50)
print("Using GPU acceleration (faster processing)" if gpu_available else "Using CPU acceleration (slower processing)")
print("=" * 50 + "\n")


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_digit_onnx_model(use_gpu):
    if not MODEL_ONNX_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_ONNX_PATH}. Run train_model.py first to generate digit_cnn.pth"
        )

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(str(MODEL_ONNX_PATH), map_location=device, weights_only=True))
    model.eval()
    return model, device


DIGIT_NET, DEVICE = load_digit_onnx_model(gpu_available)


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def corners_distance(corners1, corners2):
    """Compute max Euclidean distance between corresponding corners."""
    if corners1 is None or corners2 is None:
        return float("inf")
    dists = np.linalg.norm(corners1 - corners2, axis=1)
    return float(np.max(dists))


def preprocess_frame(frame, use_gpu):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    normalized_gray = CLAHE.apply(gray)

    if use_gpu:
        try:
            normalized_umat = cv2.UMat(normalized_gray)
            blurred_umat = cv2.GaussianBlur(normalized_umat, BLUR_KERNEL_SIZE, 0)
            binary_umat = cv2.adaptiveThreshold(
                blurred_umat,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                THRESH_BLOCK_SIZE,
                THRESH_C,
            )
            return gray, binary_umat.get(), True
        except Exception:
            pass

    blurred = cv2.GaussianBlur(normalized_gray, BLUR_KERNEL_SIZE, 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        THRESH_BLOCK_SIZE,
        THRESH_C,
    )
    return gray, binary, False


def find_grid_corners(gray, binary):
    binary_for_grid = cv2.morphologyEx(binary, cv2.MORPH_OPEN, GRID_OPEN_KERNEL)
    contours, _ = cv2.findContours(binary_for_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = gray.shape[0] * gray.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4:
            continue

        points = approximation.reshape(4, 2).astype("float32")
        return order_points(points)

    return None


def warp_binary_for_grid(binary, perspective_matrix, use_gpu):
    if use_gpu:
        try:
            warped = cv2.warpPerspective(cv2.UMat(binary), perspective_matrix, (GRID_SIZE, GRID_SIZE)).get()
            return warped, True
        except Exception:
            pass

    return cv2.warpPerspective(binary, perspective_matrix, (GRID_SIZE, GRID_SIZE)), False


def preprocess_cell(cell_binary):
    margin = int(CELL_SIZE * CELL_MARGIN_RATIO)
    cropped = cell_binary[margin:CELL_SIZE - margin, margin:CELL_SIZE - margin]
    if cropped.size == 0:
        return None

    raw_fill_ratio = float(np.count_nonzero(cropped)) / float(cropped.size)
    if raw_fill_ratio < MIN_EMPTY_PIXEL_RATIO:
        return None

    cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, CELL_CLEAN_KERNEL)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cropped, connectivity=8)
    if num_labels <= 1:
        return None

    largest_label = None
    largest_area = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        touches_border = x == 0 or y == 0 or (x + w) >= cropped.shape[1] or (y + h) >= cropped.shape[0]
        if touches_border:
            continue

        if area > largest_area:
            largest_area = area
            largest_label = label

    if largest_label is None:
        return None

    minimum_area = cropped.shape[0] * cropped.shape[1] * MIN_COMPONENT_AREA_RATIO
    if largest_area < minimum_area:
        return None

    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    if w <= 2 or h <= 6:
        return None

    fill_ratio = largest_area / float(w * h)
    if fill_ratio < MIN_COMPONENT_FILL_RATIO:
        return None

    aspect_ratio = w / float(h)
    if aspect_ratio < 0.08 or aspect_ratio > 1.4:
        return None

    mask = np.zeros_like(cropped, dtype=np.uint8)
    mask[labels == largest_label] = 255

    digit = mask[y:y + h, x:x + w]
    if digit.size == 0:
        return None

    canvas = np.zeros((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), dtype=np.uint8)
    scale = min(20.0 / w, 20.0 / h)
    resized_w = max(1, int(w * scale))
    resized_h = max(1, int(h * scale))
    resized_digit = cv2.resize(digit, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    offset_x = (MODEL_IMAGE_SIZE - resized_w) // 2
    offset_y = (MODEL_IMAGE_SIZE - resized_h) // 2
    canvas[offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = resized_digit
    return canvas


def stable_softmax(values):
    logits = np.asarray(values, dtype=np.float32).reshape(-1)
    logits = logits - np.max(logits)
    exp_values = np.exp(logits)
    denom = float(np.sum(exp_values))
    if denom <= 0:
        return np.ones_like(logits, dtype=np.float32) / float(logits.size)
    return exp_values / denom


def predict_digit_probabilities(cell_binary):
    prepared = preprocess_cell(cell_binary)
    if prepared is None:
        probs = np.zeros(10, dtype=np.float32)
        probs[0] = 1.0
        return probs

    sample = prepared.astype(np.float32) / 255.0
    blob = torch.from_numpy(sample).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = DIGIT_NET(blob)
    
    raw_output = logits.cpu().numpy().reshape(-1)

    if raw_output.size != 10:
        padded = np.zeros(10, dtype=np.float32)
        usable = min(raw_output.size, 10)
        padded[:usable] = raw_output[:usable]
        raw_output = padded

    return stable_softmax(raw_output)


def smooth_probability_history(history_entries):
    if not history_entries:
        probs = np.zeros(10, dtype=np.float32)
        probs[0] = 1.0
        return probs

    weighted_sum = np.zeros(10, dtype=np.float32)
    total_weight = 0.0
    history_length = len(history_entries)

    for index, probs in enumerate(history_entries):
        age = (history_length - 1) - index
        weight = TEMPORAL_DECAY ** age
        weighted_sum += probs * weight
        total_weight += weight

    if total_weight <= 0:
        fallback = np.zeros(10, dtype=np.float32)
        fallback[0] = 1.0
        return fallback

    smoothed = weighted_sum / total_weight
    smoothed = np.clip(smoothed, 1e-8, 1.0)
    smoothed /= float(np.sum(smoothed))
    return smoothed.astype(np.float32)


def extract_confident_grid(prob_tensor):
    grid = np.zeros((9, 9), dtype=np.int32)
    confidence = np.zeros((9, 9), dtype=np.float32)

    for row in range(9):
        for col in range(9):
            probs = prob_tensor[row, col]
            ordered = np.argsort(probs)[::-1]
            best = int(ordered[0])
            second = int(ordered[1])
            best_conf = float(probs[best])
            second_conf = float(probs[second])

            if best == 0:
                continue
            if best_conf < MIN_PREDICTION_CONFIDENCE:
                continue
            if (best_conf - second_conf) < MIN_CONFIDENCE_MARGIN:
                continue

            grid[row, col] = best
            confidence[row, col] = best_conf

    return grid, confidence


def is_valid_initial_grid(grid):
    for row in range(9):
        values = [value for value in grid[row, :] if value != 0]
        if len(values) != len(set(values)):
            return False

    for col in range(9):
        values = [value for value in grid[:, col] if value != 0]
        if len(values) != len(set(values)):
            return False

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = grid[box_row:box_row + 3, box_col:box_col + 3].flatten()
            values = [value for value in box if value != 0]
            if len(values) != len(set(values)):
                return False

    return True


def build_probability_candidates(prob_tensor):
    candidates = [[[] for _ in range(9)] for _ in range(9)]

    for row in range(9):
        for col in range(9):
            probs = prob_tensor[row, col]
            digit_probs = probs[1:]
            max_digit_prob = float(np.max(digit_probs))
            empty_prob = float(probs[0])

            if empty_prob >= EMPTY_CELL_PROBABILITY_THRESHOLD and max_digit_prob <= LOW_DIGIT_PROBABILITY_THRESHOLD:
                ranked = list(range(1, 10))
                ranked.sort(key=lambda digit: float(digit_probs[digit - 1]), reverse=True)
                candidates[row][col] = ranked
                continue

            threshold = max(ABS_DIGIT_CANDIDATE_PROB, max_digit_prob * RELATIVE_DIGIT_CANDIDATE_RATIO)
            allowed = [digit for digit in range(1, 10) if float(digit_probs[digit - 1]) >= threshold]

            if not allowed:
                allowed = [int(np.argmax(digit_probs)) + 1]

            allowed.sort(key=lambda digit: float(digit_probs[digit - 1]), reverse=True)
            candidates[row][col] = allowed[:MAX_DIGIT_CANDIDATES]

    return candidates


def solve_sudoku_with_probabilities(prob_tensor):
    candidates = build_probability_candidates(prob_tensor)
    grid = np.zeros((9, 9), dtype=np.int32)

    row_used = [set() for _ in range(9)]
    col_used = [set() for _ in range(9)]
    box_used = [set() for _ in range(9)]

    def box_index(row, col):
        return (row // 3) * 3 + (col // 3)

    def choose_next_cell():
        best_cell = None
        best_options = None
        best_entropy = float("inf")

        for row in range(9):
            for col in range(9):
                if grid[row, col] != 0:
                    continue

                valid_options = []
                for digit in candidates[row][col]:
                    box = box_index(row, col)
                    if digit in row_used[row] or digit in col_used[col] or digit in box_used[box]:
                        continue
                    valid_options.append(digit)

                if not valid_options:
                    return (row, col), []

                option_count = len(valid_options)
                certainty = float(prob_tensor[row, col, valid_options[0]])
                entropy_score = option_count - certainty

                if best_cell is None or option_count < len(best_options) or (
                    option_count == len(best_options) and entropy_score < best_entropy
                ):
                    best_cell = (row, col)
                    best_options = valid_options
                    best_entropy = entropy_score

        return best_cell, best_options

    def backtrack(filled_count):
        if filled_count == 81:
            return True

        (row, col), options = choose_next_cell()
        if row is None:
            return False
        if not options:
            return False

        options.sort(key=lambda digit: float(prob_tensor[row, col, digit]), reverse=True)

        for digit in options:
            box = box_index(row, col)
            grid[row, col] = digit
            row_used[row].add(digit)
            col_used[col].add(digit)
            box_used[box].add(digit)

            if backtrack(filled_count + 1):
                return True

            grid[row, col] = 0
            row_used[row].remove(digit)
            col_used[col].remove(digit)
            box_used[box].remove(digit)

        return False

    solved = backtrack(0)
    return solved, grid


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

corner_history = deque(maxlen=CORNER_SMOOTHING_WINDOW)
smoothed_corners_anchor = None
probability_history = [[deque(maxlen=DIGIT_SMOOTHING_WINDOW) for _ in range(9)] for _ in range(9)]
previous_signature = None
cached_solution = None
solution_display_counter = 0
fps_ema = 0.0
last_solve_latency_ms = 0.0

while True:
    frame_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    gray, binary, gpu_pipeline_active = preprocess_frame(frame, gpu_available)

    corners = find_grid_corners(gray, binary)
    if corners is None:
        corner_history.clear()
        smoothed_corners_anchor = None
        for row in range(9):
            for col in range(9):
                probability_history[row][col].clear()
        previous_signature = None
        cached_solution = None
        solution_display_counter = 0
        cv2.putText(frame, "Grid not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Real-Time Sudoku Solver", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    smoothed_corners_candidate = np.mean(np.array(corner_history), axis=0).astype(np.float32) if corner_history else None
    drift = corners_distance(corners, smoothed_corners_candidate)
    
    if drift <= CORNER_MAX_DRIFT or len(corner_history) == 0:
        corner_history.append(corners)
        smoothed_corners = np.mean(np.array(corner_history), axis=0).astype(np.float32)
        smoothed_corners_anchor = smoothed_corners
    else:
        smoothed_corners = smoothed_corners_anchor if smoothed_corners_anchor is not None else np.mean(np.array(corner_history), axis=0).astype(np.float32)
    
    smoothed_perspective = cv2.getPerspectiveTransform(smoothed_corners, WARP_DESTINATION)
    inverse_matrix = cv2.getPerspectiveTransform(WARP_DESTINATION, smoothed_corners)
    warped_binary, gpu_warp_active = warp_binary_for_grid(binary, smoothed_perspective, gpu_pipeline_active)

    probability_tensor = np.zeros((9, 9, 10), dtype=np.float32)

    for row in range(9):
        for col in range(9):
            y1 = row * CELL_SIZE
            y2 = (row + 1) * CELL_SIZE
            x1 = col * CELL_SIZE
            x2 = (col + 1) * CELL_SIZE
            cell = warped_binary[y1:y2, x1:x2]

            cell_probabilities = predict_digit_probabilities(cell)
            probability_history[row][col].append(cell_probabilities)
            smoothed_probabilities = smooth_probability_history(probability_history[row][col])
            probability_tensor[row, col] = smoothed_probabilities

    confident_grid, _ = extract_confident_grid(probability_tensor)
    filled_cells = int(np.count_nonzero(confident_grid))
    valid_clues = is_valid_initial_grid(confident_grid)

    polygon = smoothed_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon], True, (255, 0, 0), 2)

    solved = False
    solved_grid = None
    average_ocr_confidence = 0.0

    if filled_cells >= MIN_CLUES_TO_SOLVE:
        signature = tuple(np.argmax(probability_tensor, axis=2).astype(np.int8).flatten().tolist())

        if signature == previous_signature and cached_solution is not None:
            solved = True
            solved_grid = cached_solution
            solution_display_counter = SOLUTION_PERSISTENCE_FRAMES
            last_solve_latency_ms = 0.0
        else:
            solve_start = time.perf_counter()
            solved, solved_grid = solve_sudoku_with_probabilities(probability_tensor)
            last_solve_latency_ms = (time.perf_counter() - solve_start) * 1000.0
            if solved:
                previous_signature = signature
                cached_solution = solved_grid.copy()
                solution_display_counter = SOLUTION_PERSISTENCE_FRAMES
            else:
                solution_display_counter = max(0, solution_display_counter - 1)
    else:
        solution_display_counter = max(0, solution_display_counter - 1)

    if filled_cells > 0:
        digit_confidence = np.max(probability_tensor[:, :, 1:], axis=2)
        average_ocr_confidence = float(np.mean(digit_confidence[confident_grid > 0]))

    if solution_display_counter > 0 and cached_solution is not None:
        for row in range(9):
            for col in range(9):
                if confident_grid[row, col] != 0:
                    continue

                warped_point = np.array([[[col * CELL_SIZE + CELL_SIZE / 2, row * CELL_SIZE + CELL_SIZE / 2]]], dtype=np.float32)
                frame_point = cv2.perspectiveTransform(warped_point, inverse_matrix)[0][0]
                solved_text = str(int(cached_solution[row, col]))
                font_scale = 0.8
                thickness = 2
                text_size, baseline = cv2.getTextSize(solved_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = int(frame_point[0] - (text_size[0] / 2))
                text_y = int(frame_point[1] + (text_size[1] / 2) - baseline)

                cv2.putText(
                    frame,
                    solved_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                    cv2.LINE_AA,
                )

    status_color = (0, 255, 0) if valid_clues else (0, 0, 255)
    status_text = f"Detected Filled Cells Count: {filled_cells} cells"
    if not valid_clues:
        status_text += " | clues inconsistent"
    if gpu_warp_active:
        status_text += " | Using GPU"
    else:
        status_text += " | CPU"

    frame_time_ms = (time.perf_counter() - frame_start) * 1000.0
    instant_fps = 1000.0 / max(frame_time_ms, 1e-6)
    if fps_ema <= 0.0:
        fps_ema = instant_fps
    else:
        fps_ema = 0.9 * fps_ema + 0.1 * instant_fps

    metrics_text = f"FPS:{fps_ema:4.1f} | Solve:{last_solve_latency_ms:5.1f}ms | OCR:{average_ocr_confidence:.2f}"
    tuning_text = (
        f"Drift:{CORNER_MAX_DRIFT:.1f}px [ ] | Persist:{solution_display_counter}/{SOLUTION_PERSISTENCE_FRAMES} - ="
    )

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, metrics_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(frame, tuning_text, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (170, 220, 255), 2)
    cv2.imshow("Real-Time Sudoku Solver", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exiting...")
        break
cap.release()
cv2.destroyAllWindows()
