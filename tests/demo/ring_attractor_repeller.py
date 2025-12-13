"""
Interactive Ring Attractor Dynamics with Unstable Repeller

Enhanced implementation supporting single or dual ring attractors.
In dual mode, two stable limit cycles are separated by an unstable 
repeller at the midpoint, satisfying planar dynamics requirements.

Example usage:
    >>> from ring_dynamics_interactive import RingDynamicsDebug, InteractiveRingPlot
    >>> 
    >>> # Single ring attractor
    >>> ring = RingDynamicsDebug(mode='single', radius=40.0, k_radial=5.0, omega=1.0)
    >>> 
    >>> # Dual ring attractor with unstable repeller
    >>> ring = RingDynamicsDebug(mode='dual', inner_radius=30, outer_radius=45, k_radial=5.0, omega=1.0)
    >>> 
    >>> # Launch interactive plot
    >>> interactive = InteractiveRingPlot(ring)
    >>> interactive.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from typing import Tuple, List, Optional


class RingDynamicsDebug:
    """
    Standalone ring attractor dynamics for debugging.
    
    Supports two modes:
    - 'single': One stable limit cycle at radius r0
        dr/dt = -k(r - r0) + u(t)
        dtheta/dt = omega
    
    - 'dual': Two stable limit cycles at r1 and r2, separated by 
              an unstable repeller at r_mid = (r1 + r2)/2
        dr/dt = k(r - r1)(r - r_mid)(r - r2) + u(t)
        dtheta/dt = omega
        
        This cubic form ensures:
        - r1 is stable (attractor)
        - r_mid is unstable (repeller)
        - r2 is stable (attractor)
    """
    
    def __init__(
        self,
        mode: str = 'single',
        # radius: float = 30.0,
        # inner_radius: float = 15.0,
        # outer_radius: float = 45.0,
        radius: float=3.,
        inner_radius: float=3.,
        outer_radius: float=6.,
        k_radial: float = 5.0,
        omega: float = 1.0,
        dt: float = 0.1,
        initial_theta: float = 0.0,
        initial_ring: int = 0
    ):
        """
        Initialize ring attractor dynamics.
        
        Args:
            mode: 'single' or 'dual' ring attractor
            radius: Radius for single ring mode (r0)
            inner_radius: Radius of inner stable limit cycle (r1) in dual mode
            outer_radius: Radius of outer stable limit cycle (r2) in dual mode
            k_radial: Stiffness of radial dynamics (larger = faster attraction)
            omega: Angular velocity (radians per time step)
            dt: Integration time step
            initial_theta: Initial angular position (radians)
            initial_ring: Initial ring (0=inner, 1=outer) - only for dual mode
        """
        # Mode
        if mode not in ['single', 'dual']:
            raise ValueError("mode must be 'single' or 'dual'")
        self.mode = mode
        
        # Parameters
        self.r0 = radius  # Single ring radius
        self.r1 = inner_radius  # Dual ring: inner stable
        self.r2 = outer_radius  # Dual ring: outer stable
        self.r_mid = (self.r1 + self.r2) / 2.0  # Dual ring: unstable repeller
        self.k = k_radial
        self.omega = omega
        self.dt = dt
        
        # Initialize state on specified ring
        if self.mode == 'single':
            # r_init = self.r0
            r_init = 0
        else:
            # r_init = self.r1 if initial_ring == 0 else self.r2
            r_init = self.r_mid
            
        self.x = r_init * np.cos(initial_theta)
        self.y = r_init * np.sin(initial_theta)
        
        # Perturbation signal
        self.u = 0.0
        
        # History tracking
        self.time_history = [0.0]
        self.x_history = [self.x]
        self.y_history = [self.y]
        self.r_history = [r_init]
        self.theta_history = [initial_theta]
        self.u_history = [0.0]
        
        # Time counter
        self.t = 0.0
        self.step_count = 0
    
    def _compute_radial_derivative(self, r: float, u: float) -> float:
        """
        Compute radial derivative dr/dt based on mode.
        
        Args:
            r: Current radius
            u: External perturbation (radial force)
            
        Returns:
            dr/dt
        """
        if self.mode == 'single':
            # Single attractor: dr/dt = -k(r - r0) + u
            return -self.k * (r - self.r0) + u
        
        else:
            # Dual attractor with unstable repeller:
            # dr/dt = k(r - r1)(r - r_mid)(r - r2) + u
            # 
            # This cubic polynomial has the property:
            # - At r1: dr/dt ≈ 0, and d²r/dt² < 0 → stable
            # - At r_mid: dr/dt ≈ 0, and d²r/dt² > 0 → unstable
            # - At r2: dr/dt ≈ 0, and d²r/dt² < 0 → stable
            return -self.k * (r - self.r1) * (r - self.r_mid) * (r - self.r2) + u
        
    def _compute_derivatives(self, x: float, y: float, u: float) -> Tuple[float, float]:
        """
        Compute time derivatives dx/dt, dy/dt.
        
        Args:
            x, y: Current Cartesian coordinates
            u: External perturbation (radial force)
            
        Returns:
            (dx/dt, dy/dt)
        """
        # Convert to polar
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Prevent singularity at origin
        if r < 1e-6:
            r = 1e-6
            theta = 0.0
        
        # Radial dynamics (mode-dependent)
        dr_dt = self._compute_radial_derivative(r, u)
        
        # Angular dynamics: dtheta/dt = omega
        dtheta_dt = self.omega
        
        # Convert back to Cartesian derivatives
        dx_dt = dr_dt * np.cos(theta) - r * dtheta_dt * np.sin(theta)
        dy_dt = dr_dt * np.sin(theta) + r * dtheta_dt * np.cos(theta)
        
        return dx_dt, dy_dt
    
    def step(self) -> Tuple[float, float, int]:
        """
        Advance dynamics by one time step using Euler integration.
        
        Returns:
            (theta, r, ring_index)
        """
        # Compute derivatives
        dx_dt, dy_dt = self._compute_derivatives(self.x, self.y, self.u)
        
        # Euler integration
        self.x += dx_dt * self.dt
        self.y += dy_dt * self.dt
        
        # Decay perturbation exponentially
        self.u *= 0.95
        
        # Update time
        self.t += self.dt
        self.step_count += 1
        
        # Compute current state
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        if theta < 0:
            theta += 2 * np.pi
        
        # Classify ring
        if self.mode == 'single':
            ring_index = 0
        else:
            # Use repeller as boundary
            ring_index = 0 if r < self.r_mid else 1
        
        # Store history
        self.time_history.append(self.t)
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.r_history.append(r)
        self.theta_history.append(theta)
        self.u_history.append(self.u)
        
        return theta, r, ring_index
    
    def apply_perturbation(self, perturbation_strength: float) -> None:
        """
        Apply radial perturbation.
        
        Args:
            perturbation_strength: Perturbation magnitude
                Positive = push outward
                Negative = pull inward
        """
        self.u = perturbation_strength
    
    def reset(self, initial_theta: float = 0.0, initial_ring: int = 0):
        """Reset dynamics to initial state."""
        if self.mode == 'single':
            r0 = self.r0
        else:
            r0 = self.r1 if initial_ring == 0 else self.r2
            
        self.x = r0 * np.cos(initial_theta)
        self.y = r0 * np.sin(initial_theta)
        self.u = 0.0
        self.t = 0.0
        self.step_count = 0
        
        # Clear history
        self.time_history = [0.0]
        self.x_history = [self.x]
        self.y_history = [self.y]
        self.r_history = [r0]
        self.theta_history = [initial_theta]
        self.u_history = [0.0]
    
    def get_nullcline(self, r_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute nullcline (dr/dt = 0 when u = 0).
        
        Args:
            r_range: Range of radii to evaluate. If None, uses reasonable defaults.
            
        Returns:
            (r_values, dr_dt_values)
        """
        if r_range is None:
            if self.mode == 'single':
                r_min, r_max = max(0.1, self.r0 - 20), self.r0 + 20
            else:
                r_min = max(0.1, min(self.r1, self.r2) - 10)
                r_max = max(self.r1, self.r2) + 10
            r_range = np.linspace(r_min, r_max, 200)
        
        dr_dt_values = self._compute_radial_derivative(r_range, 0.0)
        return r_range, dr_dt_values


class InteractiveRingPlot:
    """
    Interactive matplotlib plot with sliders for ring attractor exploration.
    """
    
    def __init__(self, ring: Optional[RingDynamicsDebug] = None):
        """
        Initialize interactive plot.
        
        Args:
            ring: Optional pre-configured RingDynamicsDebug instance
        """
        # Initial parameters
        self.mode = 'single'
        self.radius = 40.0
        self.inner_radius = 30.0
        self.outer_radius = 45.0
        self.k_radial = 1.0
        self.omega = 1.0
        self.perturbation_time = 5.0
        self.perturbation_strength = 0.0
        self.total_time = 10.0
        self.dt = 0.0001

        # Overwrite if need be
        if ring:
            self.mode = ring.mode
            self.radius = ring.r0
            self.inner_radius = ring.r1
            self.outer_radius = ring.r2
            self.k_radial = ring.k
            self.omega = ring.omega

        # Create initial ring if not provided
        if ring is None:
            self.ring = RingDynamicsDebug(
                mode=self.mode,
                radius=self.radius,
                k_radial=self.k_radial,
                omega=self.omega
            )
        else:
            self.ring = ring
            self.mode = ring.mode
        
        self._setup_figure()
        self._run_simulation()
        self._update_plots()
    
    def _setup_figure(self):
        """Create figure with subplots and sliders."""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Create main plot area
        gs = self.fig.add_gridspec(3, 3, left=0.1, right=0.95, top=0.95, bottom=0.35,
                                   hspace=0.3, wspace=0.3)
        
        # Three main plots
        self.ax_cartesian = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_radius = self.fig.add_subplot(gs[0, 2])
        self.ax_phase = self.fig.add_subplot(gs[1, 2])
        self.ax_perturbation = self.fig.add_subplot(gs[2, :])
        
        # Slider area
        slider_left = 0.15
        slider_width = 0.35
        slider_height = 0.02
        slider_spacing = 0.03
        
        # Mode selector
        ax_mode = plt.axes([0.65, 0.25, 0.15, 0.08])
        self.radio_mode = RadioButtons(ax_mode, ('single', 'dual'), active=0)
        self.radio_mode.on_clicked(self._on_mode_change)
        
        # Create sliders
        y_pos = 0.25
        
        # Single ring radius
        ax_r0 = plt.axes([slider_left, y_pos, slider_width, slider_height])
        self.slider_r0 = Slider(ax_r0, 'Radius (single)', 1.0, 50, 
                                valinit=self.radius, valstep=1.0)
        y_pos -= slider_spacing
        
        # Inner radius
        ax_r1 = plt.axes([slider_left, y_pos, slider_width, slider_height])
        self.slider_r1 = Slider(ax_r1, 'Inner Radius', 1.0, 50.0, 
                                valinit=self.inner_radius, valstep=1.0)
        y_pos -= slider_spacing
        
        # Outer radius
        ax_r2 = plt.axes([slider_left, y_pos, slider_width, slider_height])
        self.slider_r2 = Slider(ax_r2, 'Outer Radius', 3.0, 100.0, 
                                valinit=self.outer_radius, valstep=1.0)
        y_pos -= slider_spacing
        
        # k_radial
        ax_k = plt.axes([slider_left, y_pos, slider_width, slider_height])
        self.slider_k = Slider(ax_k, 'k_radial', 0.1, 20.0, 
                               valinit=self.k_radial, valstep=0.1)
        y_pos -= slider_spacing
        
        # omega
        ax_omega = plt.axes([slider_left, y_pos, slider_width, slider_height])
        self.slider_omega = Slider(ax_omega, 'omega (ω)', 0.0, 5.0, 
                                   valinit=self.omega, valstep=0.1)
        y_pos -= slider_spacing
        
        # Perturbation time
        ax_pert_time = plt.axes([slider_left, y_pos, slider_width, slider_height])
        self.slider_pert_time = Slider(ax_pert_time, 'Perturbation Time', 0.0, 20.0, 
                                       valinit=self.perturbation_time, valstep=0.1)
        y_pos -= slider_spacing
        
        # Perturbation strength
        ax_pert_str = plt.axes([slider_left, y_pos, slider_width, slider_height])
        self.slider_pert_str = Slider(ax_pert_str, 'Perturbation Strength', -30.0, 30.0, 
                                      valinit=self.perturbation_strength, valstep=1.0)
        y_pos -= slider_spacing
        
        # Total simulation time
        ax_time = plt.axes([slider_left, y_pos, slider_width, slider_height])
        self.slider_time = Slider(ax_time, 'Total Time', 1.0, 50.0, 
                                  valinit=self.total_time, valstep=0.5)
        
        # Connect sliders to update function
        for slider in [self.slider_r0, self.slider_r1, self.slider_r2, 
                      self.slider_k, self.slider_omega, 
                      self.slider_pert_time, self.slider_pert_str, self.slider_time]:
            slider.on_changed(self._on_slider_change)
    
    def _on_mode_change(self, label):
        """Handle mode radio button changes."""
        self.mode = label
        self._on_slider_change(None)
    
    def _on_slider_change(self, val):
        """Handle slider value changes."""
        # Update parameters
        self.radius = self.slider_r0.val
        self.inner_radius = self.slider_r1.val
        self.outer_radius = self.slider_r2.val
        self.k_radial = self.slider_k.val
        self.omega = self.slider_omega.val
        self.perturbation_time = self.slider_pert_time.val
        self.perturbation_strength = self.slider_pert_str.val
        self.total_time = self.slider_time.val
        
        # Re-run simulation
        self._run_simulation()
        self._update_plots()
        self.fig.canvas.draw_idle()
    
    def _run_simulation(self):
        """Run simulation with current parameters."""
        # Create new ring with current parameters
        self.ring = RingDynamicsDebug(
            mode=self.mode,
            radius=self.radius,
            inner_radius=self.inner_radius,
            outer_radius=self.outer_radius,
            k_radial=self.k_radial,
            omega=self.omega,
            dt=0.01
        )
        
        # Calculate number of steps
        n_steps = int(self.total_time / self.ring.dt)
        perturbation_step = int(self.perturbation_time / self.ring.dt)
        
        # Run simulation
        for i in range(n_steps):
            if i == perturbation_step:
                self.ring.apply_perturbation(self.perturbation_strength)
            self.ring.step()
    
    def _update_plots(self):
        """Update all plots with current simulation data."""
        # Clear all axes
        self.ax_cartesian.clear()
        self.ax_radius.clear()
        self.ax_phase.clear()
        self.ax_perturbation.clear()
        
        # 1. Cartesian trajectory
        self._plot_cartesian()
        
        # 2. Radius vs time
        self._plot_radius_time()
        
        # 3. Phase portrait
        self._plot_phase_portrait()
        
        # 4. Perturbation signal
        self._plot_perturbation()
    
    def _plot_cartesian(self):
        """Plot Cartesian trajectory."""
        ax = self.ax_cartesian
        
        # Plot trajectory
        times = np.array(self.ring.time_history)
        scatter = ax.scatter(self.ring.x_history, self.ring.y_history, 
                           c=times, s=2, cmap='viridis', alpha=0.6)
        
        # Mark start and end
        ax.plot(self.ring.x_history[0], self.ring.y_history[0], 
               'go', markersize=8, label='Start', zorder=5)
        ax.plot(self.ring.x_history[-1], self.ring.y_history[-1], 
               'ro', markersize=8, label='End', zorder=5)
        
        # Show ring circles
        theta_circle = np.linspace(0, 2*np.pi, 100)
        
        if self.mode == 'single':
            x_ring = self.ring.r0 * np.cos(theta_circle)
            y_ring = self.ring.r0 * np.sin(theta_circle)
            ax.plot(x_ring, y_ring, 'b--', linewidth=2, alpha=0.7, 
                   label=f'Ring (r={self.ring.r0:.1f})')
        else:
            # Inner stable attractor
            x_inner = self.ring.r1 * np.cos(theta_circle)
            y_inner = self.ring.r1 * np.sin(theta_circle)
            ax.plot(x_inner, y_inner, 'b--', linewidth=2, alpha=0.7, 
                   label=f'Inner stable (r={self.ring.r1:.1f})')
            
            # Unstable repeller at midpoint
            x_mid = self.ring.r_mid * np.cos(theta_circle)
            y_mid = self.ring.r_mid * np.sin(theta_circle)
            ax.plot(x_mid, y_mid, 'gray', linewidth=1, linestyle=':', alpha=0.5, 
                   label=f'Unstable (r={self.ring.r_mid:.1f})')
            
            # Outer stable attractor
            x_outer = self.ring.r2 * np.cos(theta_circle)
            y_outer = self.ring.r2 * np.sin(theta_circle)
            ax.plot(x_outer, y_outer, 'r--', linewidth=2, alpha=0.7, 
                   label=f'Outer stable (r={self.ring.r2:.1f})')
        
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_title('Cartesian Trajectory', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    
    def _plot_radius_time(self):
        """Plot radius vs time."""
        ax = self.ax_radius
        
        ax.plot(self.ring.time_history, self.ring.r_history, 'k-', linewidth=1.5)
        
        # Show target radii
        if self.mode == 'single':
            ax.axhline(self.ring.r0, color='b', linewidth=2, linestyle='--', 
                      alpha=0.7, label=f'r={self.ring.r0:.1f}')
        else:
            ax.axhline(self.ring.r1, color='b', linewidth=2, linestyle='--', 
                      alpha=0.7, label=f'r₁={self.ring.r1:.1f} (stable)')
            ax.axhline(self.ring.r_mid, color='gray', linewidth=1, linestyle=':', 
                      alpha=0.5, label=f'r_mid={self.ring.r_mid:.1f} (unstable)')
            ax.axhline(self.ring.r2, color='r', linewidth=2, linestyle='--', 
                      alpha=0.7, label=f'r₂={self.ring.r2:.1f} (stable)')
        
        # Mark perturbation time
        ax.axvline(self.perturbation_time, color='gold', linewidth=2, 
                  linestyle=':', alpha=0.6, label='Perturbation')
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Radius', fontsize=10)
        ax.set_title('Radius vs Time', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_phase_portrait(self):
        """Plot radial phase portrait."""
        ax = self.ax_phase
        
        # Compute dr/dt from history
        dr_dt_history = []
        for i in range(len(self.ring.r_history) - 1):
            dr = self.ring.r_history[i+1] - self.ring.r_history[i]
            dr_dt = dr / self.ring.dt
            dr_dt_history.append(dr_dt)
        
        r_vals = self.ring.r_history[:-1]
        
        # Plot trajectory
        scatter = ax.scatter(r_vals, dr_dt_history, 
                           c=self.ring.time_history[:-1], 
                           s=3, cmap='viridis', alpha=0.6)
        
        # Plot nullcline
        r_range, dr_dt_null = self.ring.get_nullcline()
        ax.plot(r_range, dr_dt_null, 'k--', linewidth=2, 
               label='Nullcline (u=0)', alpha=0.7)
        
        # Mark equilibria
        if self.mode == 'single':
            ax.axvline(self.ring.r0, color='b', linewidth=2, 
                      linestyle='--', alpha=0.5)
        else:
            ax.axvline(self.ring.r1, color='b', linewidth=2, 
                      linestyle='--', alpha=0.5, label='Stable')
            ax.axvline(self.ring.r_mid, color='gray', linewidth=1, 
                      linestyle=':', alpha=0.5, label='Unstable')
            ax.axvline(self.ring.r2, color='r', linewidth=2, 
                      linestyle='--', alpha=0.5)
        
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Radius (r)', fontsize=10)
        ax.set_ylabel('dr/dt', fontsize=10)
        ax.set_title('Phase Portrait', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_perturbation(self):
        """Plot perturbation signal."""
        ax = self.ax_perturbation
        
        ax.plot(self.ring.time_history, self.ring.u_history, 
               'g-', linewidth=1.5, label='Perturbation u(t)')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(self.perturbation_time, color='gold', 
                  linewidth=2, linestyle=':', alpha=0.6)
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('u(t)', fontsize=10)
        ax.set_title('Perturbation Signal', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim([0, self.total_time])
    
    def show(self):
        """Display the interactive plot."""
        plt.show()
        

# Convenience function for quick launch
def launch_interactive(mode='single'):
    """
    Launch interactive ring attractor plot.
    
    Args:
        mode: 'single' or 'dual' ring attractor mode
    """
    if mode == 'single':
        ring = RingDynamicsDebug(mode='single')
    else:
        ring = RingDynamicsDebug(mode='dual')
    
    interactive = InteractiveRingPlot(ring)
    interactive.show()


if __name__ == "__main__":
    print("Launching interactive ring attractor debugger...")
    print("Use sliders to adjust parameters in real-time.")
    print("Toggle between single and dual ring modes with radio buttons.")
    print("\nDual mode features:")
    print("  - Two stable limit cycles (inner and outer)")
    print("  - One unstable repeller at midpoint (shown as dotted line)")
    print("  - Trajectories attracted to stable rings, repelled from midpoint")
    launch_interactive(mode='single')
    # launch_interactive(mode='dual')
