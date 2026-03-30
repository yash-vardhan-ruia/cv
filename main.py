import cv2
import numpy as np

# Enable GPU/OpenCL acceleration for OpenCV
print("=" * 50)
print("Initializing GPU Support...")
print("=" * 50)

# Check backend capabilities
print(f"OpenCV Version: {cv2.__version__}")
build_info = cv2.getBuildInformation()
print(f"OpenCL in build: {'OpenCL:                        YES' in build_info}")

gpu_available = False
try:
    cv2.ocl.setUseOpenCL(True)
    gpu_available = bool(cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL())
    print(f"OpenCL Available: {cv2.ocl.haveOpenCL()}")
    print(f"OpenCL Enabled: {cv2.ocl.useOpenCL()}")
    if gpu_available:
        print("✓ GPU acceleration ENABLED via OpenCL (NVIDIA/Direct3D backend)")
    else:
        print("✗ OpenCL runtime not active, using CPU")
except Exception as e:
    print(f"✗ GPU Access Error: {e}")
    print("Falling back to CPU.")

print("=" * 50)
if not gpu_available:
    print("Using CPU acceleration (slower processing)")
else:
    print("Using GPU acceleration (faster processing)")
print("=" * 50 + "\n")

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def build_digit_templates():
    templates = {digit: [] for digit in range(1, 10)}
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
    scales = [0.75, 0.9, 1.0]
    thicknesses = [2, 3]

    for digit in range(1, 10):
        text = str(digit)
        for font in fonts:
            for scale in scales:
                for thickness in thicknesses:
                    canvas = np.zeros((28, 28), dtype=np.uint8)
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
                    org_x = max(0, (28 - text_w) // 2)
                    org_y = max(text_h, (28 + text_h) // 2 - baseline // 2)
                    cv2.putText(canvas, text, (org_x, org_y), font, scale, 255, thickness, cv2.LINE_AA)
                    _, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    templates[digit].append(canvas)
    return templates


DIGIT_TEMPLATES = build_digit_templates()

def detect_sudoku_grid(processed_frame):
    """Detect and extract the sudoku grid from the processed frame"""
    try:
        contours, _ = cv2.findContours(processed_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Filter contours by area and shape to find the sudoku grid
        valid_contours = []
        frame_h, frame_w = processed_frame.shape[:2]
        min_area = (frame_h * frame_w) * 0.08

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue

            aspect_ratio = float(w) / h

            # Sudoku grid should be roughly square and large
            if 0.75 < aspect_ratio < 1.25:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                valid_contours.append((contour, area, (x, y, w, h), approx))
        
        if not valid_contours:
            return None
        
        # Get the largest valid contour (the sudoku grid)
        _, _, (x, y, w, h), _ = max(valid_contours, key=lambda item: item[1])
        
        # Ensure we have valid bounds
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            if y + h <= processed_frame.shape[0] and x + w <= processed_frame.shape[1]:
                extracted = processed_frame[y:y + h, x:x + w]
                if extracted.size > 0:
                    return extracted, (x, y, w, h)
    
    except Exception as e:
        print(f"Error in detect_sudoku_grid: {e}")
    
    return None

def recognize_digit_template(cell):
    """Recognize a Sudoku digit in a cell using contour isolation + template matching"""
    if cell.size == 0:
        return 0

    h, w = cell.shape[:2]
    margin = max(3, int(min(h, w) * 0.18))
    center = cell[margin:h - margin, margin:w - margin]
    if center.size == 0:
        return 0

    _, binary = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    center_area = binary.shape[0] * binary.shape[1]
    if center_area == 0 or area / center_area < 0.025:
        return 0

    x, y, bw, bh = cv2.boundingRect(largest)
    if bw <= 0 or bh <= 0:
        return 0

    digit_roi = binary[y:y + bh, x:x + bw]
    digit_28 = np.zeros((28, 28), dtype=np.uint8)
    scale = min(20.0 / bw, 20.0 / bh)
    new_w = max(1, int(bw * scale))
    new_h = max(1, int(bh * scale))
    resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    digit_28[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_digit

    best_digit = 0
    best_score = -1.0
    for digit, variants in DIGIT_TEMPLATES.items():
        for tmpl in variants:
            score = cv2.matchTemplate(digit_28, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_score = score
                best_digit = digit

    if best_score < 0.10:
        return 0
    return best_digit

def solve_sudoku(grid):
    empty_cell = find_empty_cell(grid)
    if not empty_cell:
        return True

    row, col = empty_cell

    for num in range(1, 10):
        if is_safe(grid, row, col, num):
            grid[row, col] = num

            if solve_sudoku(grid):
                return True

            grid[row, col] = 0

    return False

def find_empty_cell(grid):
    for i in range(9):
        for j in range(9):
            if grid[i, j] == 0:
                return (i, j)
    return None

def is_safe(grid, row, col, num):
    # Check row
    if num in grid[row, :]:
        return False

    # Check column
    if num in grid[:, col]:
        return False

    # Check 3x3 box
    box_start_row, box_start_col = 3 * (row // 3), 3 * (col // 3)
    if num in grid[box_start_row:box_start_row + 3, box_start_col:box_start_col + 3]:
        return False

    return True

# Rest of the code (video capture and processing)

while True:
    # Capture frame from video feed
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess the frame with GPU acceleration (OpenCL) if available
    if gpu_available:
        try:
            gray_umat = cv2.UMat(gray)
            processed_umat = cv2.GaussianBlur(gray_umat, (9, 9), 0)
            processed = processed_umat.get()
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            processed = cv2.GaussianBlur(gray, (9, 9), 0)
    else:
        processed = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Adaptive thresholding
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    # Detect the sudoku grid
    grid_result = detect_sudoku_grid(processed)
    
    if grid_result is not None:
        sudoku_grid, (grid_x, grid_y, grid_w, grid_h) = grid_result
        
        # Resize the sudoku grid to a standard size
        target_size = 450
        grid_height, grid_width = sudoku_grid.shape[:2]
        
        if grid_width == 0 or grid_height == 0:
            continue
        
        scale = min(target_size / grid_width, target_size / grid_height)
        
        new_width = int(grid_width * scale)
        new_height = int(grid_height * scale)
        
        # Make dimensions divisible by 9
        new_width = (new_width // 9) * 9
        new_height = (new_height // 9) * 9
        
        if new_width > 0 and new_height > 0:
            sudoku_grid = cv2.resize(sudoku_grid, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            continue
        
        # Split the sudoku grid into 9x9 cells
        rows, cols = 9, 9
        cell_height = new_height // rows
        cell_width = new_width // cols
        
        try:
            cells = [np.hsplit(row, cols) for row in np.vsplit(sudoku_grid, rows)]
        except Exception as e:
            print(f"Error splitting grid: {e}")
            continue
        
        # Initialize grid to store recognized digits
        sudoku_digits = np.zeros((9, 9), dtype=int)
        
        # Extract and recognize digits in each cell
        try:
            for i in range(9):
                for j in range(9):
                    cell = cells[i][j]
                    # Recognize the digit
                    digit = recognize_digit_template(cell)
                    sudoku_digits[i, j] = digit
        except Exception as e:
            print(f"Error recognizing digits: {e}")
            continue
        
        # Only solve if grid has enough filled cells
        filled_cells = np.count_nonzero(sudoku_digits)
        if filled_cells > 17:  # Valid sudoku typically has at least 17 clues
            try:
                # Make a copy for solving
                grid_copy = sudoku_digits.copy()
                solve_sudoku(grid_copy)
                
                # Calculate scale factors from original grid to frame
                scale_x = grid_w / new_width if new_width > 0 else 1
                scale_y = grid_h / new_height if new_height > 0 else 1
                
                # Overlay the solution on the frame
                for i in range(9):
                    for j in range(9):
                        if sudoku_digits[i, j] == 0 and grid_copy[i, j] != 0:
                            # This is a solved cell (was empty before)
                            cell_x = int(grid_x + j * cell_width * scale_x + cell_width * scale_x // 2)
                            cell_y = int(grid_y + i * cell_height * scale_y + cell_height * scale_y // 2)
                            cv2.putText(frame, str(grid_copy[i, j]), (cell_x - 15, cell_y + 15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error solving sudoku: {e}")
        
        # Draw the grid boundaries for visualization
        cv2.rectangle(frame, (grid_x, grid_y), (grid_x + grid_w, grid_y + grid_h), (255, 0, 0), 2)
        
        # Display detected grid information
        cv2.putText(frame, f"Detected: {filled_cells} cells", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Sudoku Solver', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()