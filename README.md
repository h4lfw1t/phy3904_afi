# PHY3904 Acoustic Field Imaging (AFI)

A Python package for processing and visualizing acoustic field measurements from dual-channel lock-in amplifier data.

## Features

- 📊 Process lock-in amplifier X/Y component data
- 📈 Calculate amplitude and phase from quadrature signals
- 🎨 Generate multiple visualization types (heatmaps, contours, 3D surfaces)
- 📁 CSV import/export for easy data handling
- 📐 Statistical analysis of acoustic fields

## Installation

### Install dependencies with Poetry

```bash
poetry install
```

### Or install dependencies with pip

```bash
pip install -r requirements.txt
```

## Quick Start

### Process Example Data

```bash
python src/main.py
```

### Use with your data

```python
from afi import AcousticFieldData, AcousticFieldVisualizer
```

### Load data (CSV with columns: x_pos, y_pos, x_comp, y_comp)

```python
data = AcousticFieldData.from_csv('your_data.csv')
```

### View statistics

```python
data.print_statistics()
```

### Create visualizations

```python
viz = AcousticFieldVisualizer(data)
viz.plot_combined(save_path='output/my_analysis.png')
```

## Project Structure

```
phy3904_afi/
├── src/afi/            # Main package
├── data/               # Input/output data
├── output/figures/     # Generated plots
└── docs/               # Documentation
```


## CSV Format

Your measurement CSV should have these columns:
- `x_pos`: X coordinate of measurement point
- `y_pos`: Y coordinate of measurement point
- `x_comp`: X (in-phase) component from lock-in amplifier
- `y_comp`: Y (quadrature) component from lock-in amplifier

## License

Academic use for PHY3904 course.