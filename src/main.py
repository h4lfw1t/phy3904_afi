"""
Main script for Acoustic Field Imaging (AFI) analysis.
Demonstrates usage with example data and provides templates for real data processing.
"""

import numpy as np
from pathlib import Path

from afi import AcousticFieldData, AcousticFieldVisualizer

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
    xx, yy = np.meshgrid(x, y)
    x_pos = xx.flatten()
    y_pos = yy.flatten()

    # Simulate acoustic field (interference pattern)
    x_comp = np.sin(0.5 * x_pos) * np.cos(0.5 * y_pos)
    y_comp = np.cos(0.5 * x_pos) * np.sin(0.5 * y_pos)

    # Add realistic noise
    x_comp += np.random.normal(0, 0.05, x_comp.shape)
    y_comp += np.random.normal(0, 0.05, y_comp.shape)

    return AcousticFieldData(x_pos, y_pos, x_comp, y_comp)

def generate_theoretical_model(grid_size: int = 9,
                               f: float = 2000.0) -> AcousticFieldData:
    """
    Generate first-order theoretical model of the acoustic field.

    :param grid_size: Size of the grid (grid_size x grid_size points)
    :type grid_size: int
    :param f: Frequency of the wave (Hz)
    :type f: float, optional
    :return: AcousticFieldData object with theoretical data
    :rtype: AcousticFieldData
    """
    print(f"Generating theoretical model for a {grid_size}x{grid_size} grid...")

    # Acoustic Constants
    v = 343.0                   # Speed of sound in m/s
    k = 2 * np.pi * f / v       # Wave number in rad/m

    # Emitter position parameters
    z0 = 0.40       # cm
    center_x, center_y = 4.5, 4.5

    # Create grid
    x = np.linspace(0, 9, grid_size)
    y = np.linspace(0, 9, grid_size)
    xx, yy = np.meshgrid(x, y)
    x_pos = xx.flatten()
    y_pos = yy.flatten()

    # Calculate true 3D distance from emitter to microphone
    dx_m = (x_pos - center_x) * 0.01  # Convert cm to m
    dy_m = (y_pos - center_y) * 0.01  # Convert cm to m

    r_3d = np.sqrt(dx_m**2 + dy_m**2 + z0**2)

    # Calculate amplitude (1/r)
    amp = 1.0 / r_3d
    normalized_amp = amp / np.max(amp)

    # Calculate phase (k*r)
    phase = k * r_3d

    # Simulate lock-in amplifier output
    x_comp = normalized_amp * np.cos(phase)
    y_comp = normalized_amp * np.sin(phase)

    return AcousticFieldData(x_pos, y_pos, x_comp, y_comp)

def process_example_data():
    """Run complete analysis pipeline on example data."""
    # Create output directories
    output_dir = OUT_DIR / 'example'
    data_dir = DATA_DIR / 'example/processed'
    figures_dir = output_dir / 'example/figures'

    for directory in [output_dir, data_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

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

def process_theoretical_data():
    """Run complete analysis pipeline on theoretical model."""
    # Create output directories
    output_dir = OUT_DIR / 'theoretical'
    data_dir = DATA_DIR / 'theoretical/processed'
    figures_dir = output_dir / 'theoretical/figures'

    for directory in [output_dir, data_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Generate theoretical model
    data = generate_theoretical_model(grid_size=9)

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
    process_theoretical_data()

    # Example 3: Process real data
    for file in DATA_DIR.glob('raw/*.csv'):
        print("\n" + "="*60)
        print(f"PROCESSING REAL DATA: {file.name}")
        print("="*60)
        process_real_data(file, output_name=file.stem)