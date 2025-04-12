import json

with open("StrokePredictor.ipynb", "r") as f:
    notebook = json.load(f)

code_cells = [
    cell["source"]
    for cell in notebook["cells"]
    if cell["cell_type"] == "code"
]

with open("extracted_code.py", "w") as f:
    for cell in code_cells:
        f.write("".join(cell) + "\n")