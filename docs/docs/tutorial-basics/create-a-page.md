---
sidebar_position: 1
---

# 📘 Formatting your Input Data

## Preparing your Data  

Greetings, Cell Shape Researcher! 🧪 Before you embark on your analysis, it's crucial to ensure that your data is formatted correctly for our application.

### 📜 Format & Structure

Your data should be presented in a text file with xy coordinates for each cell:

- Each xy coordinate pair should be separated by a space.
- Different cells should be distinguished with a line break.

**Example:**

```
x1 y1
x2 y2
x3 y3
...

x1 y1
x2 y2
```

Each pair denotes a point on the cell's boundary. The blank line signifies the start of a new cell's data.

### 🖥️ Data Parsing Procedure

Our application utilizes a specialized function to process your data:

```python
def parse_coordinates(file_path):
    # ... (function definition as you provided) ...
```

This function segregates the cells based on line breaks and delineates individual points using spaces.

## Preparing Your Coordinate Data

Welcome to the Cell Shape Analysis App! To ensure a smooth and efficient analysis, it's essential that your cell shape coordinate data is in the correct format. Let's delve into how you can achieve this.

#### 📜 Desired Format & Structure

Your data should be structured in the following manner:

- Each xy coordinate pair should be separated by a space.
- Different cells should be distinguished with a line break.

**Example:**

```
x1 y1
x2 y2
x3 y3
...

x1 y1
x2 y2
```

### 🖥️ Helper Functions to Convert Your Data

If your data isn't already in this format, don't worry! Below are a few Python helper functions to assist you in converting your data:

1. **From List of Lists to Desired Format**:
If you have your data in a list of lists (where each list represents a cell's coordinates), use this function:

```python
def from_lists_to_format(cells):
    formatted_data = ""
    for cell in cells:
        for coord in cell:
            formatted_data += f"{coord[0]} {coord[1]}\n"
        formatted_data += "\n"
    return formatted_data
```

Usage:

```python
data = [
    [[1,2], [3,4], [5,6]],
    [[7,8], [9,10]]
]
formatted_data = from_lists_to_format(data)
print(formatted_data)
```

2. **From CSV to Desired Format**:
If your data is in a CSV format where each row represents a coordinate and each cell is separated by a new row, use this function:

```python
import csv

def from_csv_to_format(csv_path):
    formatted_data = ""
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            formatted_data += f"{row[0]} {row[1]}\n"
        formatted_data += "\n"
    return formatted_data
```

Usage:

```python
formatted_data = from_csv_to_format("path_to_your_file.csv")
print(formatted_data)
```

Once you've transformed your data using one of the helper functions above, you can save the output to a text file or directly input it into our Cell Shape Analysis App.

🔍 **Tip**: Always double-check your formatted data to ensure there aren't any discrepancies. Proper data preparation is the foundation of accurate analysis. Happy Analyzing! 🎉


### ❗ Common Mistakes & Corrections

1. **Missing Line Breaks**: Ensure each cell's data is separated by a line break. This distinction is vital for accurate analysis.
2. **Incorrect Delimiters**: Use a space to demarcate the x and y coordinates. Other delimiters will lead to parsing errors.
3. **Extraneous Data**: Only include the xy coordinates in the file. Any additional data will be disregarded.

### 🚀 Ready to Proceed?

With your data formatted correctly, you're poised to unlock insights into the world of cell shapes. Ensure adherence to the guidelines for optimal results.

🤓 **Tip**: Cells might be small, but their details are profound. Happy Analyzing! 🎉
