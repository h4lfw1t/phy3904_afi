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

    def __init__(self,
                 x_positions: np.ndarray,
                 y_positions: np.ndarray,
                 x_component: np.ndarray,
                 y_component: np.ndarray,
                 z_position: float = 0.4,
                 center_x: float = None,
                 center_y: float = None,
                 normalize_to_center: bool = True,
                 frequency: float = None):
        """
        Initialize acoustic field data.

        Parameters:
        -----------
        x_positions : np.ndarray
            X coordinates of measurement points in cm
        y_positions : np.ndarray
            Y coordinates of measurement points in cm
        x_component : np.ndarray
            X (in-phase) component from lock-in amplifier
        y_component : np.ndarray
            Y (quadrature) component from lock-in amplifier
        z_position: float, optional
            Z coordinate of measurement points in meters (default: 0.4 m)
        center_x: float, optional
            X coordinate of emitter center in meters. Defaults to geometric center.
        center_y: float, optional
            Y coordinate of emitter center in meters. Defaults to geometric center.
        normalize_to_center: bool, optional
            If True, normalize amplitude to value at spatial center (0, 0).
            If False, use raw amplitude. Default = True
        frequency: float, optional
            Frequency of the acoustic field in Hz. If None, theoretical_phase is not calculated.
        """
        self.x_pos = np.asarray(x_positions)
        self.y_pos = np.asarray(y_positions)
        self.x_comp = np.asarray(x_component)
        self.y_comp = np.asarray(y_component)
        self.z_pos = z_position
        self.frequency = frequency

        # Validate data
        if not (len(self.x_pos) == len(self.y_pos) == len(self.x_comp) == len(self.y_comp)):
            raise ValueError("All input arrays must have the same length")

        # Dynamically determine the center index based on proximity to origin (0,0) by default
        # or proximity to the geometric center for example data.
        if center_x is None or center_y is None:
            # If no center is forced, find the origin point. Real data is centered at 0,0.
            center_idx = np.argmin(self.x_pos**2 + self.y_pos**2)
            self.center_x = self.x_pos[center_idx] * 0.01
            self.center_y = self.y_pos[center_idx] * 0.01
        else:
            self.center_x = center_x
            self.center_y = center_y

        self.center_idx = np.argmin(
            (self.x_pos * 0.01 - self.center_x)**2 + 
            (self.y_pos * 0.01 - self.center_y)**2
        )

        # Calculate derived quantities
        self.amplitude = np.sqrt(self.x_comp**2 + self.y_comp**2)
        self.phase = np.mod(np.arctan2(self.y_comp, self.x_comp), 2 * np.pi)  # Phase in radians [0, 2pi)

        if normalize_to_center:
            norm_factor = self.amplitude[self.center_idx]
            if norm_factor > 0:
                self.amplitude /= norm_factor

        # Calculate unwrapped phase
        self.unwrapped_phase = self._unwrap_phase_2d(self.phase)

        # Calculate relative phase
        self.relative_phase = self._calculate_relative_phase(self.unwrapped_phase)

    def _unwrap_phase_2d(self, phase: np.ndarray) -> np.ndarray:
        """
        Unwraps phase over a 2D spatial grid to prevent artificial
        discontinuities at row boundaries.
        """
        nx = len(np.unique(self.x_pos))
        ny = len(np.unique(self.y_pos))

        # Verify that the flattened vectors map perfectly to a 2D rectangle
        if nx * ny != len(phase):
            raise ValueError("Spatial coordinates do not map to a complete rectangular grid.")

        # Reshape the 1D phase array into the proper 2D spatial matrix
        phase_matrix = phase.reshape((ny, nx))

        # Execute the 2D unwrapping algorithm
        unwrapped_matrix = unwrap_phase(phase_matrix)

        # Flatten the matrix back to a 1D array
        return unwrapped_matrix.flatten()

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
        # Find phase at the acoustic center
        center_phase = phase[self.center_idx]
        return phase - center_phase

    @classmethod
    def from_csv(cls, filepath: str, frequency: float = None) -> 'AcousticFieldData':
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

        # Ensure spatial arrays are perfectly sorted into [y_pos, x_pos] grid format
        df = df.sort_values(by=['y_pos', 'x_pos'])

        return cls(
            x_positions=df['x_pos'].values,
            y_positions=df['y_pos'].values,
            x_component=df['x_comp'].values,
            y_component=df['y_comp'].values,
            frequency=frequency
        )

    @classmethod
    def from_theoretical_model(cls, x_pos: np.ndarray, y_pos: np.ndarray, frequency: float, center_x: float = None, center_y: float = None, z_pos: float = 0.40) -> 'AcousticFieldData':
        """
        Generate theoretical acoustic field on arbitrary points (e.g., experimental positions).

        Parameters:
        -----------
        x_pos : np.ndarray
            X positions of points (same units as measurement)
        y_pos : np.ndarray
            Y positions of points
        frequency : float
            Frequency in Hz

        Returns:
        --------
        AcousticFieldData object
        """
        v = 343.0                   # Speed of sound in m/s
        k = 2 * np.pi * frequency / v  # Wavenumber in rad/m

        # Emitter position parameters
        z0 = z_pos                   # Distance from emitter to surface in m

        # Determine center if not specified explicitly
        if center_x is None or center_y is None:
            center_idx = np.argmin(x_pos**2 + y_pos**2)
            cx_m = center_x if center_x is not None else x_pos[center_idx] * 0.01
            cy_m = center_y if center_y is not None else y_pos[center_idx] * 0.01
        else:
            cx_m = center_x
            cy_m = center_y

        # Convert positions to meters, relative to acoustic center
        dx_m = (x_pos * 0.01) - cx_m
        dy_m = (y_pos * 0.01) - cy_m

        # Calculate 3D distance from emitter to microphone
        r_3d = np.sqrt(dx_m**2 + dy_m**2 + z0**2)

        # Calculate amplitude and phase
        amp = 1 / r_3d
        phase = k * r_3d

        x_comp = amp * np.cos(phase)
        y_comp = amp * np.sin(phase)

        return cls(x_pos, y_pos, x_comp, y_comp, center_x=center_x, center_y=center_y, z_position=z_pos, frequency=frequency)

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