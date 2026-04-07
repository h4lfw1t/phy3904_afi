r"""
Main script for Acoustic Field Imaging (AFI) analysis.
Demonstrates usage with example data and provides templates for real data processing.

TO RUN: 
poetry env use "C:\Users\lucie\AppData\Local\Programs\Python\Python312\python.exe"
poetry install 
poetry run python src/main.py 
"""

import numpy as np
# pathlib provides object-oriented tools for handling file paths 
# Path class allows representing file or directory path as an object
from pathlib import Path 

from afi import AcousticFieldData, AcousticFieldVisualizer

#__file__ is a special Python variable that containes the path of the current Python file
# Path(__file__) turns that string into a Path object from pathlib 
#.parent.parent goes up two directories from the current file 
# / 'out' and / 'data': using / on a Path object appends a subdirectory or filename
OUT_DIR = Path(__file__).parent.parent / 'out'
DATA_DIR = Path(__file__).parent.parent / 'data'

def generate_example_data(grid_size: int = 9) -> AcousticFieldData:
    """
    Generate synthetic acoustic field data for testing.

    :param grid_size Size of the grid (grid_size x grid_size points)
    :type grid_size: int
    :return: AcousticFieldData object with synthetic data
    :rtype: AcousticFieldData
    """
    print(f"Generating example data for a {grid_size}x{grid_size} grid...")

    # Create grid
    x = np.linspace(0, 9, grid_size)
    y = np.linspace(0, 9, grid_size)
    xx, yy = np.meshgrid(x, y) #turns into 2D arrays 
    x_pos = xx.flatten() #turn back into 1D arrays (for functions like griddata to work)
    y_pos = yy.flatten()

    # Simulate acoustic field (interference pattern)
    x_comp = np.sin(0.5 * x_pos) * np.cos(0.5 * y_pos)
    y_comp = np.cos(0.5 * x_pos) * np.sin(0.5 * y_pos)

    # Add realistic noise
    x_comp += np.random.normal(0, 0.05, x_comp.shape) #(mean,std,shape)
    y_comp += np.random.normal(0, 0.05, y_comp.shape)

    return AcousticFieldData(x_pos, y_pos, x_comp, y_comp)

def generate_theoretical_model_from_points(x_pos: np.ndarray, y_pos: np.ndarray, f: float) -> AcousticFieldData:
    """
    Generate theoretical acoustic field on arbitrary points (e.g., experimental positions).

    :param x_pos: x positions of points (same units as measurement)
    :param y_pos: y positions of points
    :param f: frequency in Hz
    :return: AcousticFieldData object with x_comp, y_comp
    """
    v = 343.0
    k = 2 * np.pi * f / v
    z0 = 0.40  # emitter height in meters

    # Convert positions to meters if they are in cm
    dx_m = x_pos * 0.01
    dy_m = y_pos * 0.01

     # 3D distance from emitter
    r_3d = np.sqrt(dx_m**2 + dy_m**2 + z0**2)
    A0 = 1.0
    amp = A0 * (z0 / r_3d)

    phase = k * r_3d

    x_comp = amp * np.cos(phase)
    y_comp = amp * np.sin(phase)

    return AcousticFieldData(x_pos, y_pos, x_comp, y_comp)

def process_example_data():
    """Run complete analysis pipeline on example data."""
    # Create output directories
    output_dir = OUT_DIR / 'example'
    data_dir = DATA_DIR / 'example/processed'
    figures_dir = output_dir / 'example/figures'

    for directory in [output_dir, data_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True) #mkdir makes directory

    # Generate or load data
    data = generate_example_data(grid_size=9)

    # Save raw processed data
    data.to_csv(data_dir / 'example_processed.csv')

    # Print statistics
    data.print_statistics()

    # Create visualizer
    viz = AcousticFieldVisualizer(data)

    # Generate all visualizations
    print("\nGenerating visualizations...")
    viz.plot_combined(save_path=figures_dir / 'combined_view.png')
    viz.plot_amplitude_heatmap(save_path=figures_dir / 'amplitude_heatmap.png')
    viz.plot_phase_heatmap(save_path=figures_dir / 'phase_heatmap.png')
    viz.plot_amplitude_contours(save_path=figures_dir / 'amplitude_contours.png')
    viz.plot_phase_contours(save_path=figures_dir / 'phase_contours.png')
    viz.plot_amplitude_3d_surface(save_path=figures_dir / 'amplitude_3d_surface.png')
    viz.plot_phase_3d_surface(save_path=figures_dir / 'phase_3d_surface.png')

    print(f"\n✓ Analysis complete! Results saved to '{output_dir}' directory.")
    print(f"✓ Processed data saved to '{data_dir}' directory.")
    print(f"✓ Figures saved to '{figures_dir}' directory.")

def process_theoretical_data(csv_path: str, output_name: str = 'analysis'):
    """
    Run complete analysis pipeline on theoretical model, normalized to experimental data.

    Parameters:
    -----------
    csv_path : str
        Path to CSV file with experimental data
    output_name : str
        Base name for output files
    """
    # Create output directories
    output_dir = OUT_DIR / 'theoretical'
    data_dir = DATA_DIR / 'theoretical/processed'
    figures_dir = output_dir / 'theoretical/figures'

    for directory in [output_dir, data_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Load experimental data
    exp_data = AcousticFieldData.from_csv(csv_path)

    # Generate theoretical model
    data = generate_theoretical_model_from_points(exp_data.x_pos,exp_data.y_pos,f=3000.0)

    # --- Normalize theoretical amplitudes to experimental data ---
    # Find experimental center (closest point to (0,0))
    exp_center_idx = np.argmin(exp_data.x_pos**2 + exp_data.y_pos**2)
    amp_experimental_center = np.sqrt(
        exp_data.x_comp[exp_center_idx]**2 +
        exp_data.y_comp[exp_center_idx]**2
    )

    # Find theoretical center (closest point to experimental center)
    dist2 = (data.x_pos - exp_data.x_pos[exp_center_idx])**2 + \
            (data.y_pos - exp_data.y_pos[exp_center_idx])**2
    theo_center_idx = np.argmin(dist2)
    amp_theoretical_center = np.sqrt(
        data.x_comp[theo_center_idx]**2 + data.y_comp[theo_center_idx]**2
    )

    # Scale theoretical data
    scale_factor = amp_experimental_center / amp_theoretical_center
    data.x_comp *= scale_factor
    data.y_comp *= scale_factor
    # -------------------------------------------------------------

    # Save raw processed data
    data.to_csv(data_dir / 'theoretical_processed.csv')

    # Print statistics
    data.print_statistics()

    # Create visualizer
    viz = AcousticFieldVisualizer(data)

    # Generate all visualizations
    print("\nGenerating visualizations...")
    viz.plot_combined(save_path=figures_dir / 'combined_view.png')
    viz.plot_amplitude_heatmap(save_path=figures_dir / 'amplitude_heatmap.png')
    viz.plot_phase_heatmap(save_path=figures_dir / 'phase_heatmap.png')
    viz.plot_unwrapped_phase_heatmap(save_path=figures_dir / 'unwrapped_phase_heatmap.png')
    viz.plot_theoretical_phase_heatmap(save_path=figures_dir / 'theoretical_phase_heatmap.png')
    viz.plot_relative_phase_heatmap(save_path=figures_dir / 'relative_phase_heatmap.png')
    viz.plot_amplitude_contours(save_path=figures_dir / 'amplitude_contours.png')
    viz.plot_phase_contours(save_path=figures_dir / 'phase_contours.png')
    viz.plot_amplitude_3d_surface(save_path=figures_dir / 'amplitude_3d_surface.png')
    viz.plot_phase_3d_surface(save_path=figures_dir / 'phase_3d_surface.png')

    print(f"\n✓ Analysis complete! Results saved to '{output_dir}' directory.")
    print(f"✓ Processed data saved to '{data_dir}' directory.")
    print(f"✓ Figures saved to '{figures_dir}' directory.")

def process_real_data(csv_path: str, output_name: str = 'analysis'):
    """
    Process real measurement data from CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to CSV file with columns: x_pos, y_pos, x_comp, y_comp
    output_name : str
        Base name for output files
    """
    # Create output directories
    output_dir = OUT_DIR
    data_dir = DATA_DIR / 'processed'
    figures_dir = output_dir / 'figures'

    for directory in [output_dir, data_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {csv_path}...")
    data = AcousticFieldData.from_csv(csv_path)

    # Save processed data
    data.to_csv(data_dir / f'{output_name}_processed.csv')

    # Print statistics
    data.print_statistics()

    # Create visualizer and generate plots
    viz = AcousticFieldVisualizer(data)

    print("\nGenerating visualizations...")
    viz.plot_combined(save_path=figures_dir / f'{output_name}_combined.png')
    viz.plot_amplitude_heatmap(save_path=figures_dir / f'{output_name}_amplitude.png')
    viz.plot_phase_heatmap(save_path=figures_dir / f'{output_name}_phase.png')
    viz.plot_unwrapped_phase_heatmap(save_path=figures_dir / f'{output_name}_unwrapped_phase.png')
    viz.plot_theoretical_phase_heatmap(save_path=figures_dir / f'{output_name}_theoretical_phase.png')
    viz.plot_relative_phase_heatmap(save_path=figures_dir / f'{output_name}_relative_phase.png')
    viz.plot_amplitude_contours(save_path=figures_dir / f'{output_name}_amplitude_contours.png')
    viz.plot_phase_contours(save_path=figures_dir / f'{output_name}_phase_contours.png')
    viz.plot_amplitude_3d_surface(save_path=figures_dir / f'{output_name}_amplitude_3d.png')
    viz.plot_phase_3d_surface(save_path=figures_dir / f'{output_name}_phase_3d.png')

    print(f"\n✓ Analysis complete! Results saved to '{output_dir}' directory.")

# Only run the code below if this file is being run directly, not imported 
if __name__ == '__main__':
    # # Example 1: Process synthetic example data
    # print("="*60)
    # print("EXAMPLE: Processing synthetic data")
    # print("="*60)
    # process_example_data()

    # Example 2: Process theoretical model
    print("\n" + "="*60)
    print("EXAMPLE: Processing theoretical model")
    print("="*60)
    process_theoretical_data(f'{DATA_DIR}/raw/PHY3904_BaEP_3kHz.csv', output_name='3khz')

    # Example 3: Process real data
    print("\n" + "="*60)
    print("PROCESSING REAL DATA")
    print("="*60)
    process_real_data(f'{DATA_DIR}/raw/PHY3904_BaEP_3kHz.csv', output_name='3khz')