"""
Main script for Acoustic Field Imaging (AFI) analysis.
Demonstrates usage with example data and provides templates for real data processing.
"""

import numpy as np
import re
from pathlib import Path 

from afi import AcousticFieldData, AcousticFieldVisualizer

OUT_DIR = Path(__file__).parent.parent / 'out'
DATA_DIR = Path(__file__).parent.parent / 'data'

def generate_example_data(grid_size: int = 9) -> AcousticFieldData:
    """
    Generate synthetic acoustic field data for testing.

    :param grid_size: Size of the grid (grid_size x grid_size points)
    :return: AcousticFieldData object with synthetic data
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

def process_example_data():
    """Run complete analysis pipeline on example data."""
    # Create output directories
    output_dir = OUT_DIR / 'example'
    data_dir = DATA_DIR / 'example/processed'
    figures_dir = output_dir / 'figures'

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
    viz.plot_unwrapped_phase_3d_surface(save_path=figures_dir / 'unwrapped_phase_3d_surface.png')

    print(f"\n✓ Analysis complete! Results saved to '{output_dir}' directory.")
    print(f"✓ Processed data saved to '{data_dir}' directory.")
    print(f"✓ Figures saved to '{figures_dir}' directory.")

def process_theoretical_data(csv_path: str,
                             f: float,
                             output_name: str):
    """
    Process theoretical model data based on experimental positions.

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
    figures_dir = output_dir / 'figures'

    for directory in [output_dir, data_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Load experimental data
    exp_data = AcousticFieldData.from_csv(csv_path)

    # Generate theoretical model
    data = AcousticFieldData.from_theoretical_model(
        exp_data.x_pos,
        exp_data.y_pos,
        frequency=f,
    )

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
    viz.plot_relative_phase_heatmap(save_path=figures_dir / 'relative_phase_heatmap.png')
    viz.plot_amplitude_contours(save_path=figures_dir / 'amplitude_contours.png')
    viz.plot_phase_contours(save_path=figures_dir / 'phase_contours.png')
    viz.plot_amplitude_3d_surface(save_path=figures_dir / 'amplitude_3d_surface.png')
    viz.plot_phase_3d_surface(save_path=figures_dir / 'phase_3d_surface.png')
    viz.plot_unwrapped_phase_3d_surface(save_path=figures_dir / 'unwrapped_phase_3d_surface.png')
    viz.plot_relative_phase_3d_surface(save_path=figures_dir / 'relative_phase_3d_surface.png')

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
    viz.plot_relative_phase_heatmap(save_path=figures_dir / f'{output_name}_relative_phase.png')
    viz.plot_amplitude_contours(save_path=figures_dir / f'{output_name}_amplitude_contours.png')
    viz.plot_phase_contours(save_path=figures_dir / f'{output_name}_phase_contours.png')
    viz.plot_amplitude_3d_surface(save_path=figures_dir / f'{output_name}_amplitude_3d.png')
    viz.plot_phase_3d_surface(save_path=figures_dir / f'{output_name}_phase_3d.png')
    viz.plot_unwrapped_phase_3d_surface(save_path=figures_dir / f'{output_name}_unwrapped_phase_3d.png')
    viz.plot_relative_phase_3d_surface(save_path=figures_dir / f'{output_name}_relative_phase_3d.png')

    print(f"\n✓ Analysis complete! Results saved to '{output_dir}' directory.")


if __name__ == '__main__':
    raw_files = list(DATA_DIR.glob('raw/*.csv'))
    
    if not raw_files:
        print(f"No CSV files found in {DATA_DIR / 'raw'}")
    
    for filepath in raw_files:
        filename = filepath.name
        
        # Extract frequency from filename (e.g., '3kHz' -> 3000.0)
        freq_match = re.search(r'(\d+)kHz', filename)
        if not freq_match:
            print(f"Skipping {filename}: Could not extract frequency from name.")
            continue
            
        freq = float(freq_match.group(1)) * 1000.0
        output_name = filename.replace('.csv', '')
        
        print("\n" + "="*60)
        print(f"PROCESSING REAL DATA: {filename}")
        print("="*60)
        process_real_data(str(filepath), output_name=output_name)
        
        print("\n" + "="*60)
        print(f"PROCESSING THEORETICAL MODEL: {filename}")
        print("="*60)
        process_theoretical_data(str(filepath), f=freq, output_name=output_name)