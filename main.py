import cv2
import numpy as np
from collections import Counter

# Enable GPU acceleration for OpenCV
print("Initializing GPU...")
print(f"CUDA Available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cv2.cuda.setDevice(0)
    print("GPU initialized successfully")
else:
    print("No CUDA GPU found, using CPU")

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def detect_sudoku_grid(processed_frame):
    """Detect and extract the sudoku grid from the processed frame"""
    contours, _ = cv2.findContours(processed_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Filter contours by area and aspect ratio to find the sudoku grid
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10000:  # Minimum area for sudoku grid
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Sudoku grid should be roughly square (0.7 to 1.3 aspect ratio)
        if 0.7 < aspect_ratio < 1.3:
            valid_contours.append((contour, area))
    
    if not valid_contours:
        return None
    
    # Get the largest valid contour (the sudoku grid)
    largest_contour, _ = max(valid_contours, key=lambda x: x[1])
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return processed_frame[y:y + h, x:x + w], (x, y, w, h)

def recognize_digit_template(cell):
    """Recognize digits using template matching and morphological operations"""
    if cell.size == 0:
        return 0
    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(cell, 127, 255, cv2.THRESH_BINARY)
    
    # Count non-zero pixels
    non_zero_count = cv2.countNonZero(binary)
    total_pixels = cell.size
    fill_ratio = non_zero_count / total_pixels
    
    # If fill ratio is too low, cell is empty
    if fill_ratio < 0.05:
        return 0
    
    # Resize cell to standard size for analysis
    resized = cv2.resize(cell, (32, 32))
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the cell
    contours, _ = cv2.findContours(morph.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0
    
    # Get the largest contour (the digit)
    digit_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(digit_contour)
    
    if contour_area < 50:
        return 0
    
    # Calculate shape descriptors
    x, y, w, h = cv2.boundingRect(digit_contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Use Hu moments for digit classification
    hu_moments = cv2.HuMoments(digit_contour).flatten()
    
    # Simple heuristic based on multiple features
    area_ratio = contour_area / (32 * 32)
    
    # Classify digit based on characteristics
    if area_ratio < 0.1:
        return 0  # Too small
    elif aspect_ratio > 2.0 or aspect_ratio < 0.3:
        return 0  # Too elongated, probably noise
    
    # Use pixel density for rough digit estimation
    white_pixels = cv2.countNonZero(resized)
    
    # Map pixel count to digit ranges (1-9)
    if white_pixels < 50:
        return 0
    elif white_pixels < 120:
        return 1
    elif white_pixels < 180:
        return 2
    elif white_pixels < 250:
        return 3
    elif white_pixels < 320:
        return 4
    elif white_pixels < 400:
        return 5
    elif white_pixels < 480:
        return 6
    elif white_pixels < 560:
        return 7
    elif white_pixels < 640:
        return 8
    else:
        return 9

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

# Rest of the code (video capture and processing) remains the same.

while True:
    # Capture frame from video feed
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess the frame
    processed = cv2.GaussianBlur(gray, (9, 9), 0)
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the processed frame
    contours, _ = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the largest contour (presumably the Sudoku grid)
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the Sudoku grid from the frame
        x, y, w, h = cv2.boundingRect(largest_contour)
        sudoku_grid = processed[y:y + h, x:x + w]

        # # Resize the Sudoku grid to a fixed size (if necessary)
        # sudoku_grid = cv2.resize(sudoku_grid, (450, 450))

        # Determine the desired size for the Sudoku grid
        grid_size = 450

        # # Resize the Sudoku grid to a size that can be evenly divided into 9 cells
        # grid_height, grid_width = sudoku_grid.shape[:2]
        # resized_height = (grid_height // grid_size) * grid_size
        # resized_width = (grid_width // grid_size) * grid_size
        # sudoku_grid = cv2.resize(sudoku_grid, (resized_width, resized_height))

        # Resize the Sudoku grid to a size that can be evenly divided into 9 cells
        grid_height, grid_width = sudoku_grid.shape[:2]
        resized_height = int(grid_height / grid_size) * grid_size
        resized_width = int(grid_width / grid_size) * grid_size
        # sudoku_grid = cv2.resize(sudoku_grid, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        if resized_width > 0 and resized_height > 0:
            sudoku_grid = cv2.resize(sudoku_grid, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        else:
            continue



        # Prepare the Sudoku grid for digit extraction
        sudoku_grid = cv2.copyMakeBorder(sudoku_grid, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

        # Ensure dimensions are divisible by 9
        rows, cols = 9, 9
        grid_height, grid_width = sudoku_grid.shape[:2]
        
        # Trim to nearest multiple of 9
        grid_height = (grid_height // rows) * rows
        grid_width = (grid_width // cols) * cols
        sudoku_grid = sudoku_grid[:grid_height, :grid_width]

        # Split the Sudoku grid into individual cells
        cell_height = grid_height // rows
        cell_width = grid_width // cols

        cells = [np.hsplit(row, cols) for row in np.vsplit(sudoku_grid, rows)]


        # Initialize an empty grid to store the Sudoku digits
        sudoku_digits = np.zeros((9, 9), dtype=int)

        # Extract and recognize digits in each cell of the Sudoku grid
        for i in range(9):
            for j in range(9):
                cell = cells[i][j]
                cell = cv2.bitwise_not(cell)  # Invert the cell to make digits black
                
                # Perform digit recognition on the cell
                digit = recognize_digit(cell)
                sudoku_digits[i, j] = digit

        # Solve the Sudoku puzzle
        solve_sudoku(sudoku_digits)

        # Overlay the solution on the frame
        cell_size = frame.shape[0] // 9
        for i in range(9):
            for j in range(9):
                digit = sudoku_digits[i, j]
                if digit != 0:
                    cell_x = x + j * cell_size
                    cell_y = y + i * cell_size
                    cv2.putText(frame, str(digit), (cell_x + 30, cell_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('Real-Time Sudoku Solver', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()