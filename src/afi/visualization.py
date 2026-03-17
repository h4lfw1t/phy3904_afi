"""
Visualization module for acoustic field data.
Provides various plotting functions for analyzing acoustic field measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Tuple, Optional

from .data import AcousticFieldData


class AcousticFieldVisualizer:
    """Visualization tools for acoustic field data."""

    def __init__(self, data: AcousticFieldData):
        """
        Initialize visualizer with acoustic field data.

        Parameters:
        -----------
        data : AcousticFieldData
            Acoustic field data object
        """
        self.data = data

    def _prepare_grid(self, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare interpolated grid for visualization."""
        # Create regular grid
        xi = np.linspace(self.data.x_pos.min(), self.data.x_pos.max(), resolution)
        yi = np.linspace(self.data.y_pos.min(), self.data.y_pos.max(), resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate amplitude
        amp_grid = griddata(
            (self.data.x_pos, self.data.y_pos),
            self.data.amplitude,
            (xi_grid, yi_grid),
            method='cubic'
        )

        # Interpolate phase
        phase_grid = griddata(
            (self.data.x_pos, self.data.y_pos),
            self.data.phase,
            (xi_grid, yi_grid),
            method='cubic'
        )

        return xi_grid, yi_grid, amp_grid, phase_grid

    def plot_amplitude_heatmap(self, figsize: Tuple[int, int] = (10, 8),
                               cmap: str = 'viridis', show_points: bool = True,
                               save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of the amplitude field.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap name
        show_points : bool
            Whether to show measurement points
        save_path : str, optional
            Path to save the figure
        """
        xi_grid, yi_grid, amp_grid, _ = self._prepare_grid()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.pcolormesh(xi_grid, yi_grid, amp_grid, cmap=cmap, shading='auto')

        # Optionally show measurement points
        if show_points:
            ax.scatter(self.data.x_pos, self.data.y_pos, c='red',
                       s=20, marker='x', alpha=0.5, label='Measurement points')
            ax.legend()

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Acoustic Field Amplitude Map')
        ax.set_aspect('equal')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Amplitude heatmap saved to {save_path}")

        plt.show()

    def plot_phase_heatmap(self, figsize: Tuple[int, int] = (10, 8),
                           cmap: str = 'twilight', show_points: bool = True,
                           save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of the phase field.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap name (twilight is good for phase)
        show_points : bool
            Whether to show measurement points
        save_path : str, optional
            Path to save the figure
        """
        xi_grid, yi_grid, _, phase_grid = self._prepare_grid()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.pcolormesh(xi_grid, yi_grid, phase_grid, cmap=cmap,
                           shading='auto', vmin=-np.pi, vmax=np.pi)

        # Optionally show measurement points
        if show_points:
            ax.scatter(self.data.x_pos, self.data.y_pos, c='black',
                       s=20, marker='x', alpha=0.5, label='Measurement points')
            ax.legend()

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Acoustic Field Phase Map')
        ax.set_aspect('equal')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Phase (radians)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Phase heatmap saved to {save_path}")

        plt.show()

    def plot_contours(self, num_levels: int = 15, figsize: Tuple[int, int] = (10, 8),
                      cmap: str = 'viridis', save_path: Optional[str] = None) -> None:
        """
        Create contour plot of the amplitude field.

        Parameters:
        -----------
        num_levels : int
            Number of contour levels
        figsize : tuple
            Figure size
        cmap : str
            Colormap name
        save_path : str, optional
            Path to save the figure
        """
        xi_grid, yi_grid, amp_grid, _ = self._prepare_grid()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot filled contours
        contourf = ax.contourf(xi_grid, yi_grid, amp_grid, levels=num_levels, cmap=cmap)

        # Plot contour lines
        contour = ax.contour(xi_grid, yi_grid, amp_grid, levels=num_levels,
                             colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8)

        # Show measurement points
        ax.scatter(self.data.x_pos, self.data.y_pos, c='red',
                   s=20, marker='x', alpha=0.5, label='Measurement points')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Acoustic Field Amplitude Contours')
        ax.set_aspect('equal')
        ax.legend()

        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Amplitude')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Contour plot saved to {save_path}")

        plt.show()

    def plot_3d_surface(self, figsize: Tuple[int, int] = (12, 9),
                        cmap: str = 'viridis', save_path: Optional[str] = None) -> None:
        """
        Create 3D surface plot of the amplitude field.

        Parameters:
        -----------
        figsize : tuple
            Figure size
        cmap : str
            Colormap name
        save_path : str, optional
            Path to save the figure
        """
        xi_grid, yi_grid, amp_grid, _ = self._prepare_grid()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(xi_grid, yi_grid, amp_grid, cmap=cmap,
                               edgecolor='none', alpha=0.9)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Amplitude')
        ax.set_title('Acoustic Field 3D Surface')

        fig.colorbar(surf, ax=ax, shrink=0.5, label='Amplitude')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D surface plot saved to {save_path}")

        plt.show()

    def plot_combined(self, figsize: Tuple[int, int] = (16, 6),
                      save_path: Optional[str] = None) -> None:
        """
        Create a combined figure with amplitude and phase maps.

        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        xi_grid, yi_grid, amp_grid, phase_grid = self._prepare_grid()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Amplitude plot
        im1 = ax1.pcolormesh(xi_grid, yi_grid, amp_grid, cmap='viridis', shading='auto')
        ax1.scatter(self.data.x_pos, self.data.y_pos, c='red',
                    s=15, marker='x', alpha=0.5)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Amplitude')
        ax1.set_aspect('equal')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Amplitude')

        # Phase plot
        im2 = ax2.pcolormesh(xi_grid, yi_grid, phase_grid, cmap='twilight',
                             shading='auto', vmin=-np.pi, vmax=np.pi)
        ax2.scatter(self.data.x_pos, self.data.y_pos, c='black',
                    s=15, marker='x', alpha=0.5)
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Phase')
        ax2.set_aspect('equal')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Phase (radians)')

        plt.suptitle('Acoustic Field Analysis', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined plot saved to {save_path}")

        plt.show()