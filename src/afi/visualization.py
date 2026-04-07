"""
Visualization module for acoustic field data.
Provides various plotting functions for analyzing acoustic field measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from scipy.interpolate import griddata
from typing import Tuple, Optional

from .data import AcousticFieldData

class DataType(Enum):
    """Enumeration for acoustic field data types"""
    AMPLITUDE = 'amplitude'
    PHASE = 'phase'
    UNWRAPPED_PHASE = 'unwrapped_phase'
    RELATIVE_PHASE = 'relative_phase'

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

    def _prepare_grid(self, resolution: int = 100):
        """Prepare interpolated grid for visualization, using complex interpolation for phases."""

        # Create regular grid
        xi = np.linspace(self.data.x_pos.min(), self.data.x_pos.max(), resolution)
        yi = np.linspace(self.data.y_pos.min(), self.data.y_pos.max(), resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate amplitude (normal)
        amp_grid = griddata(
            (self.data.x_pos, self.data.y_pos),
            self.data.amplitude,
            (xi_grid, yi_grid),
            method='cubic'
        )

        # Interpolate wrapped phase using complex representation
        phase_complex = np.exp(1j * self.data.phase)
        phase_complex_grid = griddata(
            (self.data.x_pos, self.data.y_pos),
            phase_complex,
            (xi_grid, yi_grid),
            method='linear'  # linear avoids cubic artifacts
        )
        phase_grid = np.mod(np.angle(phase_complex_grid), 2 * np.pi)

        # Interpolate unwrapped phase as a normal scalar field
        unwrapped_phase_grid = griddata(
            (self.data.x_pos, self.data.y_pos),
            self.data.unwrapped_phase,
            (xi_grid, yi_grid),
            method='cubic'
        )

        # Interpolate relative phase normally
        relative_phase_grid = griddata(
            (self.data.x_pos, self.data.y_pos),
            self.data.relative_phase,
            (xi_grid, yi_grid),
            method='cubic'
        )

        return xi_grid, yi_grid, amp_grid, phase_grid, unwrapped_phase_grid, relative_phase_grid

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
        return self._plot_heatmap(DataType.AMPLITUDE, figsize, cmap, show_points, save_path)

    def plot_phase_heatmap(self, figsize: Tuple[int, int] = (10, 8),
                           cmap: str = 'cividis', show_points: bool = True,
                           save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of the phase field.

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
        return self._plot_heatmap(DataType.PHASE, figsize, cmap, show_points, save_path)

    def plot_unwrapped_phase_heatmap(self, figsize: Tuple[int, int] = (10, 8),
                                     cmap: str = 'magma', show_points: bool = True,
                                     save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of the unwrapped phase field.

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
        return self._plot_heatmap(DataType.UNWRAPPED_PHASE, figsize, cmap, show_points, save_path)

    def plot_relative_phase_heatmap(self, figsize: Tuple[int, int] = (10, 8),
                                    cmap: str = 'inferno', show_points: bool = True,
                                    save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of the relative phase field.

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
        return self._plot_heatmap(DataType.RELATIVE_PHASE, figsize, cmap, show_points, save_path)

    def _plot_heatmap(self, data_type: DataType = DataType.AMPLITUDE,
                      figsize: Tuple[int, int] = (10, 8),
                      cmap: str = 'cividis', show_points: bool = True,
                      save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of the acoustic field data.

        :param data_type: DataType
            Which data to plot (AMPLITUDE or PHASE)
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap name
        show_points : bool
            Whether to show measurement points
        save_path : str, optional
            Path to save the figure
        """
        config = {
            DataType.AMPLITUDE: {
                'data': None,
                'default_cmap': 'viridis',
                'title': 'Acoustic Field Amplitude Map',
                'cbar_label': 'Amplitude',
                'point_color': 'red',
                'vmin': None,
                'vmax': None,
            },
            DataType.PHASE: {
                'data': None,
                'default_cmap': 'cividis',
                'title': 'Acoustic Field Phase Map',
                'cbar_label': 'Phase (radians)',
                'point_color': 'black',
                'vmin': 0,
                'vmax': 2 * np.pi,
            },
            DataType.UNWRAPPED_PHASE: {
                'data': None,
                'default_cmap': 'magma',
                'title': 'Unwrapped Acoustic Field Phase Map',
                'cbar_label': 'Unwrapped Phase (radians)',
                'point_color': 'black',
                'vmin': None,
                'vmax': None,
            },
            DataType.RELATIVE_PHASE: {
                'data': None,
                'default_cmap': 'inferno',
                'title': 'Relative Acoustic Field Phase Map',
                'cbar_label': 'Relative Phase (radians)',
                'point_color': 'black',
                'vmin': 0,
                'vmax': None,
            }
        }

        # Get grids
        xi_grid, yi_grid, amp_grid, phase_grid, unwrapped_phase_grid, relative_phase_grid = self._prepare_grid()
        config[DataType.AMPLITUDE]['data'] = amp_grid
        config[DataType.PHASE]['data'] = phase_grid
        config[DataType.UNWRAPPED_PHASE]['data'] = unwrapped_phase_grid
        config[DataType.RELATIVE_PHASE]['data'] = relative_phase_grid

        # Get data type settings
        settings = config[data_type]
        cmap = cmap or settings['default_cmap']

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.pcolormesh(xi_grid, yi_grid, settings['data'], cmap=cmap,
                           shading='auto', vmin=settings['vmin'], vmax=settings['vmax'])

        # Optionally show measurement points
        if show_points:
            ax.scatter(self.data.x_pos, self.data.y_pos, c=settings['point_color'],
                       s=20, marker='x', alpha=0.5, label='Measurement points')
            ax.legend()

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(settings['title'])
        ax.set_aspect('equal')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(settings['cbar_label'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{settings['title']} saved to {save_path}")

        plt.show()

    def plot_amplitude_contours(self, num_levels: int = 15,
                               figsize: Tuple[int, int] = (10, 8),
                               cmap: str = 'viridis',
                               save_path: Optional[str] = None) -> None:
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
        return self._plot_contours(DataType.AMPLITUDE, num_levels, figsize, cmap, save_path)

    def plot_phase_contours(self, num_levels: int = 15,
                               figsize: Tuple[int, int] = (10, 8),
                               cmap: str = 'cividis',
                               save_path: Optional[str] = None) -> None:
        """
        Create contour plot of the phase field.

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
        return self._plot_contours(DataType.PHASE, num_levels, figsize, cmap, save_path)

    def _plot_contours(self, data_type: DataType = DataType.AMPLITUDE,
                      num_levels: int = 15, figsize: Tuple[int, int] = (10, 8),
                      cmap: str = 'viridis', save_path: Optional[str] = None) -> None:
        """
        Create contour plot of the acoustic field data.

        :param data_type: DataType
            Which data to plot (AMPLITUDE or PHASE)
        :param num_levels: int
            Number of contour levels
        :param figsize: tuple
            Figure size
        :param cmap: str
            Colormap name
        :param save_path: str, optional
            Path to save the figure
        :return:
        """
        config = {
            DataType.AMPLITUDE: {
                'data': None,
                'default_cmap': 'viridis',
                'title': 'Acoustic Field Amplitude Contours',
                'cbar_label': 'Amplitude',
                'point_color': 'red',
                'vmin': None,
                'vmax': None,
            },
            DataType.PHASE: {
                'data': None,
                'default_cmap': 'cividis',
                'title': 'Acoustic Field Phase Contours',
                'cbar_label': 'Phase (radians)',
                'point_color': 'red',
                'vmin': 0,
                'vmax': 2 * np.pi,
            }
        }

        # Get grids
        xi_grid, yi_grid, amp_grid, phase_grid, _, _ = self._prepare_grid()
        config[DataType.AMPLITUDE]['data'] = amp_grid
        config[DataType.PHASE]['data'] = phase_grid

        # Get data type settings
        settings = config[data_type]
        num_levels = num_levels or settings['num_levels']
        cmap = cmap or settings['default_cmap']

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot filled contours
        contourf = ax.contourf(xi_grid, yi_grid, settings['data'], levels=num_levels, cmap=cmap)

        # Plot contour lines
        contour = ax.contour(xi_grid, yi_grid, settings['data'], levels=num_levels,
                             colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8)

        # Show measurement points
        ax.scatter(self.data.x_pos, self.data.y_pos, c=settings['point_color'],
                   s=20, marker='x', alpha=0.5, label='Measurement points')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(settings['title'])
        ax.set_aspect('equal')
        ax.legend()

        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label(settings['cbar_label'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Contour plot saved to {save_path}")

        plt.show()

    def plot_amplitude_3d_surface(self, figsize: Tuple[int, int] = (12, 9),
                                  cmap: str = 'viridis',
                                  save_path: Optional[str] = None) -> None:
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
        return self._plot_3d_surface(DataType.AMPLITUDE, figsize, cmap, save_path)

    def plot_phase_3d_surface(self, figsize: Tuple[int, int] = (12, 9),
                                  cmap: str = 'cividis',
                                  save_path: Optional[str] = None) -> None:
        """
        Create 3D surface plot of the phase field.

        Parameters:
        -----------
        figsize : tuple
            Figure size
        cmap : str
            Colormap name
        save_path : str, optional
            Path to save the figure
        """
        return self._plot_3d_surface(DataType.PHASE, figsize, cmap, save_path)

    def plot_unwrapped_phase_3d_surface(self, figsize: Tuple[int, int] = (12, 9),
                                  cmap: str = 'magma',
                                  save_path: Optional[str] = None) -> None:
        """
        Create 3D surface plot of the unwrapped phase field.

        Parameters:
        -----------
        figsize : tuple
            Figure size
        cmap : str
            Colormap name
        save_path : str, optional
            Path to save the figure
        """
        return self._plot_3d_surface(DataType.UNWRAPPED_PHASE, figsize, cmap, save_path)

    def plot_relative_phase_3d_surface(self, figsize: Tuple[int, int] = (12, 9),
                                  cmap: str = 'inferno',
                                  save_path: Optional[str] = None) -> None:
        """
        Create 3D surface plot of the relative phase field.

        Parameters:
        -----------
        figsize : tuple
            Figure size
        cmap : str
            Colormap name
        save_path : str, optional
            Path to save the figure
        """
        return self._plot_3d_surface(DataType.RELATIVE_PHASE, figsize, cmap, save_path)

    def _plot_3d_surface(self, data_type: DataType = DataType.AMPLITUDE,
                         figsize: Tuple[int, int] = (12, 9),
                         cmap: str = 'viridis',
                         save_path: Optional[str] = None) -> None:
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
        config = {
            DataType.AMPLITUDE: {
                'data': None,
                'default_cmap': 'viridis',
                'title': 'Acoustic Field Amplitude Surface',
                'cbar_label': 'Amplitude',
            },
            DataType.PHASE: {
                'data': None,
                'default_cmap': 'cividis',
                'title': 'Acoustic Field Phase Surface',
                'cbar_label': 'Phase (radians)',
            },
            DataType.UNWRAPPED_PHASE: {
                'data': None,
                'default_cmap': 'magma',
                'title': 'Unwrapped Acoustic Field Phase Surface',
                'cbar_label': 'Unwrapped Phase (radians)',
            },
            DataType.RELATIVE_PHASE: {
                'data': None,
                'default_cmap': 'inferno',
                'title': 'Relative Acoustic Field Phase Surface',
                'cbar_label': 'Relative Phase (radians)',
            }
        }

        # Get grids
        xi_grid, yi_grid, amp_grid, phase_grid, unwrapped_phase_grid, relative_phase_grid = self._prepare_grid()
        config[DataType.AMPLITUDE]['data'] = amp_grid
        config[DataType.PHASE]['data'] = phase_grid
        config[DataType.UNWRAPPED_PHASE]['data'] = unwrapped_phase_grid
        config[DataType.RELATIVE_PHASE]['data'] = relative_phase_grid

        # Get data type settings
        settings = config[data_type]
        cmap = cmap or settings['default_cmap']

        # Create plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(xi_grid, yi_grid, settings['data'], cmap=cmap,
                               edgecolor='none', alpha=0.9)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel(settings['cbar_label'])
        ax.set_title(settings['title'])

        fig.colorbar(surf, ax=ax, shrink=0.5, label=settings['cbar_label'])

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
        xi_grid, yi_grid, amp_grid, phase_grid, _, _ = self._prepare_grid()

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
        im2 = ax2.pcolormesh(xi_grid, yi_grid, phase_grid, cmap='cividis',
                             shading='auto', vmin=0, vmax=2*np.pi)
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