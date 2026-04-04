"""
Data processing module for acoustic field measurements.
Handles loading, processing, and statistical analysis of lock-in amplifier data.
"""

import numpy as np
import pandas as pd
from skimage.restoration import unwrap_phase
from typing import Dict


class AcousticFieldData:
    """Handles acoustic field data processing and analysis."""

    def __init__(self, x_positions: np.ndarray, y_positions: np.ndarray,
                 x_component: np.ndarray, y_component: np.ndarray,
                 z_position: float = 0.4, center_x: float = 0.045,
                 center_y: float = 0.045):
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
        z_position: float, optional
            Z coordinate of measurement points (default: 0.4 m)
        center_x: float, optional
            X coordinate of emitter center (default: 0.045 m)
        center_y: float, optional
            Y coordinate of emitter center (default: 0.045 m)
        """
        self.x_pos = np.asarray(x_positions)
        self.y_pos = np.asarray(y_positions)
        self.x_comp = np.asarray(x_component)
        self.y_comp = np.asarray(y_component)
        self.z_pos = z_position
        self.center_x = center_x
        self.center_y = center_y

        # Validate data
        if not (len(self.x_pos) == len(self.y_pos) == len(self.x_comp) == len(self.y_comp)):
            raise ValueError("All input arrays must have the same length")

        # Calculate derived quantities
        self.amplitude = np.sqrt(self.x_comp**2 + self.y_comp**2)
        self.phase = np.arctan2(self.y_comp, self.x_comp)  # Phase in radians

        # Calculate unwrapped phase
        self.unwrapped_phase = self._unwrap_phase_2d(self.phase)

        # Calculate theoretical phase model
        self.theoretical_phase = self._calculate_theoretical_phase()

        # Calculate relative phase
        self.relative_phase = self._calculate_relative_phase(self.phase)

    def _unwrap_phase_2d(self, phase: np.ndarray) -> np.ndarray:
        """
                Unwraps phase over a 2D spatial grid to prevent artificial
                discontinuities at row boundaries.
                """
        # 1. Ascertain the grid dimensions from the coordinate vectors
        nx = len(np.unique(self.x_pos))
        ny = len(np.unique(self.y_pos))

        # Verify that the flattened vectors map perfectly to a 2D rectangle
        if nx * ny != len(phase):
            raise ValueError("Spatial coordinates do not map to a complete rectangular grid.")

        # 2. Reshape the 1D phase array into the proper 2D spatial matrix.
        # When using np.meshgrid with the default 'xy' indexing,
        # the resulting shape corresponds to (ny, nx).
        phase_matrix = phase.reshape((ny, nx))

        # 3. Execute the 2D unwrapping algorithm
        unwrapped_matrix = unwrap_phase(phase_matrix)

        # 4. Flatten the matrix back to a 1D array to maintain parity
        # with our x_pos and y_pos vectors
        return unwrapped_matrix.flatten()

    def _calculate_theoretical_phase(self, f: float = 2000.0) -> np.ndarray:
        k = 2 * np.pi * f / 343.0  # Wave number in rad/m

        dx_m = self.x_pos * 0.01
        dy_m = self.y_pos * 0.01

        r = np.sqrt(dx_m**2 + dy_m**2 + self.z_pos**2)

        return k * (r - self.z_pos)

    def _calculate_relative_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        Determines the relative phase by comparing phase values radially with the center of the acoustic field.

        Parameters:
        -----------
        phase : np.ndarray
            Phase values in radians

        Returns:
        --------
        np.ndarray
            Relative phase values in radians
        """
        # Find phase where x and y positions are zero
        center_mask = (self.x_pos == 0) & (self.y_pos == 0)

        if not np.any(center_mask):
            raise ValueError("No measurement points at the center (x_pos=0, y_pos=0)")

        center_phase = phase[center_mask][0]

        return phase - center_phase

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