"""
Data processing module for acoustic field measurements.
Handles loading, processing, and statistical analysis of lock-in amplifier data.
"""

import numpy as np
import pandas as pd
from typing import Dict


class AcousticFieldData:
    """Handles acoustic field data processing and analysis."""

    def __init__(self, x_positions: np.ndarray, y_positions: np.ndarray,
                 x_component: np.ndarray, y_component: np.ndarray):
        """
        Initialize acoustic field data.

        Parameters:
        -----------
        x_positions : np.ndarray
            X coordinates of measurement points
        y_positions : np.ndarray
            Y coordinates of measurement points
        x_component : np.ndarray
            X (in-phase) component from lock-in amplifier
        y_component : np.ndarray
            Y (quadrature) component from lock-in amplifier
        """
        self.x_pos = np.asarray(x_positions)
        self.y_pos = np.asarray(y_positions)
        self.x_comp = np.asarray(x_component)
        self.y_comp = np.asarray(y_component)

        # Validate data
        if not (len(self.x_pos) == len(self.y_pos) == len(self.x_comp) == len(self.y_comp)):
            raise ValueError("All input arrays must have the same length")

        # Calculate derived quantities
        self.amplitude = np.sqrt(self.x_comp**2 + self.y_comp**2)
        self.phase = np.arctan2(self.y_comp, self.x_comp)  # Phase in radians

    @classmethod
    def from_csv(cls, filepath: str) -> 'AcousticFieldData':
        """
        Load data from CSV file.

        Expected columns: x_pos, y_pos, x_comp, y_comp

        Parameters:
        -----------
        filepath : str
            Path to CSV file

        Returns:
        --------
        AcousticFieldData object
        """
        df = pd.read_csv(filepath)
        required_cols = ['x_pos', 'y_pos', 'x_comp', 'y_comp']

        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        return cls(
            x_positions=df['x_pos'].values,
            y_positions=df['y_pos'].values,
            x_component=df['x_comp'].values,
            y_component=df['y_comp'].values
        )

    def to_csv(self, filepath: str) -> None:
        """Save processed data to CSV file."""
        df = pd.DataFrame({
            'x_pos': self.x_pos,
            'y_pos': self.y_pos,
            'x_comp': self.x_comp,
            'y_comp': self.y_comp,
            'amplitude': self.amplitude,
            'phase': self.phase
        })
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistical properties of the acoustic field."""
        return {
            'amplitude_mean': np.mean(self.amplitude),
            'amplitude_std': np.std(self.amplitude),
            'amplitude_max': np.max(self.amplitude),
            'amplitude_min': np.min(self.amplitude),
            'phase_mean': np.mean(self.phase),
            'phase_std': np.std(self.phase),
            'num_points': len(self.x_pos)
        }

    def print_statistics(self) -> None:
        """Print statistical summary of the acoustic field."""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("ACOUSTIC FIELD STATISTICS")
        print("="*50)
        print(f"Number of measurement points: {stats['num_points']}")
        print(f"\nAmplitude:")
        print(f"  Mean:   {stats['amplitude_mean']:.4f}")
        print(f"  Std:    {stats['amplitude_std']:.4f}")
        print(f"  Max:    {stats['amplitude_max']:.4f}")
        print(f"  Min:    {stats['amplitude_min']:.4f}")
        print(f"\nPhase (radians):")
        print(f"  Mean:   {stats['phase_mean']:.4f}")
        print(f"  Std:    {stats['phase_std']:.4f}")
        print("="*50 + "\n")