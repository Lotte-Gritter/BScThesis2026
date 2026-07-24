# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:42:30 2025

@author: Jacob Seifert, janbr
"""



#%% Run before other skimage import-commands to avoid error involving skimage.color.rgb2gray

from skimage.color import rgb2gray

#%%

import scipy as sc
from scipy.interpolate import interpn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


import skimage as skimage
from skimage import data, transform

from tqdm import tqdm
import os


#%%

#Experimental wavelength, optical eff. distance between camera-chip & object
exp_wavelength = 532e-9;
exp_z_dist = 24.29e-3 #24.06e-3;#24.39e-3;#27.3e-3;#26.824e-3;

#Params used in simulation: padding-factor for FFT, number of pixels of object, number of pixels used to crop sim. diffraction-pattern
sim_pad_factor = 2;
sim_N_pix_obj = 1024;
sim_N_crop = 839;

#Number of pixels & pixel-size of the (Basler)-cam
cam_N_pix = 1024;
cam_pix_size = 6.9e-6;

#Pixel size of the object & diffuser used in the (Fourier-transform based) simulation
sim_obj_pix_size = sim_N_crop / (sim_pad_factor * sim_N_pix_obj) * exp_wavelength * exp_z_dist / (cam_N_pix * cam_pix_size);




#%%

class Correlated_Speckle_Imaging(nn.Module):
    def __init__(self, params, speckle_data):
        """
        Initialize the Correlated_Speckle_Imaging class.

        Args:
            params (dict): Dictionary containing:
                - device: torch.device
                - wavelength (float): Wavelength in meters
                - prop_dist (float): Propagation distance in meters
                - pixel_size_object (float): Pixel size in object plane in meters
                - num_pixels_object (int): Number of pixels in object plane
                - M (float): Magnification factor
                - speckle_NA (float): Numerical aperture for speckle generation
                - support_size (float): Object support size in meters
                - pad_factor (int): Padding factor for propagation
                - crop_size (int): Crop size for propagated field in pixels
                - step_size (float): Physical step size in meters (pre-demagnification)
                - num_steps (int): Number of scanning steps
            speckle_data (dict): Dictionary containing scanning positions.
        """
        super().__init__()
        self.params = params
        self.device = params["device"]
        self.wavelength = params["wavelength"]
        self.total_photon_number = params.get("total_photon_number", None)
        self.prop_dist = params["prop_dist"]
        self.pixel_size_object = params["pixel_size_object"]
        self.num_pixels_object = (
            params["num_pixels_object"],
            params["num_pixels_object"],
        )
        self.M = params["M"]
        self.speckle_NA = params["speckle_NA"]
        self.support_size = params["support_size"]
        self.pad_factor = params.get("pad_factor", 2)
        self.crop_size = params.get("crop_size", 239)
        scan_pattern_np = speckle_data.get("scanning_positions")  # shape [num_steps, 2]
        self.scanning_positions = (
            torch.tensor(scan_pattern_np, dtype=torch.float32, device=self.device)
            / self.M
        )

        # Object plane grid
        y = (
            torch.arange(-self.num_pixels_object[0] / 2, self.num_pixels_object[0] / 2)
            * self.pixel_size_object
        )
        x = (
            torch.arange(-self.num_pixels_object[1] / 2, self.num_pixels_object[1] / 2)
            * self.pixel_size_object
        )
        self.yy_object, self.xx_object = torch.meshgrid(y, x, indexing="ij")
        self.yy_object = self.yy_object.to(self.device)
        self.xx_object = self.xx_object.to(self.device)
        # Compute support mask based on support_size
        r = torch.sqrt(self.yy_object ** 2 + self.xx_object ** 2)
        
        self.eta_object_gen = 0.6; #Additional scaling factor for simulated object
        
        self.support_mask = (r <= self.eta_object_gen * self.support_size / 2).float().to(self.device)

        # Initialize diffuser and object
        self.diffuser_seed = 1234
        self.initialize_diffuser(self.total_photon_number)
        self.object = nn.Parameter(
            torch.ones(
                self.num_pixels_object, dtype=torch.complex64, device=self.device
            )
        )
        self.diffraction_patterns = None
        
        

    def initialize_diffuser(self, total_photon_number=None):
        """
        Initialize the diffuser so that it can be shifted by all scanning positions
        without wrapping. We compute the bounding box of the scanning positions
        (already scaled to the object plane), add that to the object size, and
        then generate a sufficiently large 2D speckle field.
        """
        # Object size in physical units
        obj_height_m = self.num_pixels_object[1] * self.pixel_size_object
        obj_width_m = self.num_pixels_object[0] * self.pixel_size_object

        # Scanning pattern bounding box (in the object plane)
        min_x = torch.min(self.scanning_positions[:, 1])
        max_x = torch.max(self.scanning_positions[:, 1])
        min_y = torch.min(self.scanning_positions[:, 0])
        max_y = torch.max(self.scanning_positions[:, 0])

        # Total required physical size in each dimension
        total_width_m = obj_width_m + (max_x - min_x)
        total_height_m = obj_height_m + (max_y - min_y)

        # Convert physical size to pixel counts
        nx = int(torch.ceil(total_width_m / self.pixel_size_object))
        ny = int(torch.ceil(total_height_m / self.pixel_size_object))

        # Keep them at least as big as the object itself
        margin = 50  # extra margin frame to ensure no wrapping
        nx = max(nx + margin, self.num_pixels_object[1])
        ny = max(ny + margin, self.num_pixels_object[0])

        self.num_pixels_diffuser = (ny, nx)
        print(f"Diffuser total number of pixels: {self.num_pixels_diffuser}")
        self.diffuser, self.diffuser_mask = self.generate_diffuser()
        self.diffuser = self.diffuser.to(self.device)
        self.diffuser_mask = self.diffuser_mask.to(self.device)
        if total_photon_number is not None:
            current_photon_number = torch.sum(torch.abs(self.diffuser) ** 2)
            self.diffuser = self.diffuser * torch.sqrt(
                total_photon_number / current_photon_number
            )
            scaled_photon_number = torch.sum(torch.abs(self.diffuser) ** 2)
            print(f"Scaled diffuser photon number: {scaled_photon_number.item():.0f}")

    def generate_diffuser(self):
        """
        Generate the initial speckle diffuser field as a complex64 tensor.
        We apply a random phase within the support determined by speckle_NA.
        No additional shifting/padding is done here, just the raw speckle creation.
        """
        # Reproducible random seed
        torch.manual_seed(self.diffuser_seed)

        # Frequency grid for the diffuser
        ny, nx = self.num_pixels_diffuser
        fy = torch.fft.fftfreq(ny, d=self.pixel_size_object, device=self.device)
        fx = torch.fft.fftfreq(nx, d=self.pixel_size_object, device=self.device)
        fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")

        # Maximum spatial frequency from the numerical aperture
        f_max = 0.2 * self.speckle_NA / self.wavelength

        # Build frequency mask (1 inside the circle of radius f_max, 0 outside)
        mask = (torch.sqrt(fyy ** 2 + fxx ** 2) <= f_max).float()

        # Random phases in [0, 2π)
        phase = 2 * np.pi * torch.rand_like(mask)

        # Construct complex field in frequency domain
        ft_field = mask * torch.exp(1j * phase)

        # Inverse FFT to get the diffuser in real space
        diffuser = torch.fft.ifftn(ft_field)

        # Return as complex64
        return diffuser.to(torch.complex64), mask

    def generate_object(self):
        """Generate a binary amplitude object with circular support."""
        eta_object_gen = self.eta_object_gen;
        support_pixels = int(eta_object_gen * self.support_size / self.pixel_size_object)
        cameraman = transform.resize(
            data.camera(), (support_pixels, support_pixels), anti_aliasing=True
        )
        object_field = torch.tensor(cameraman, dtype=torch.float32, device=self.device)
        object_field = (object_field > 0.5).float()  # Binarize

        # Apply circular mask
        r = torch.sqrt(self.yy_object ** 2 + self.xx_object ** 2)
        mask = (r <= eta_object_gen * self.support_size / 2).float()
        full_object = torch.zeros(self.num_pixels_object, device=self.device)
        start = (self.num_pixels_object[0] - support_pixels) // 2
        end = start + support_pixels
        full_object[start:end, start:end] = object_field * mask[start:end, start:end]

        # store in attribute
        self.object.data = full_object
        return full_object
    
    def generate_phase_object(self):
        """Generate a complex object with amplitude and phase contrast."""
        eta_object_gen = self.eta_object_gen;
        support_pixels = int(eta_object_gen * self.support_size / self.pixel_size_object)
        
        # Load and resize camera man for amplitude
        camera_man_resized = transform.resize(data.camera(), (support_pixels, support_pixels), anti_aliasing=True)
        amplitude = 0.2 + 0.8 * torch.tensor(camera_man_resized, dtype=torch.float32, device=self.device)
        
        # Load and resize astronaut for phase
        astronaut_gray = rgb2gray(data.astronaut())
        astronaut_resized = transform.resize(astronaut_gray, (support_pixels, support_pixels), anti_aliasing=True)
        phase = np.pi * torch.tensor(astronaut_resized, dtype=torch.float32, device=self.device)
        
        # Create complex object field
        object_field = amplitude * torch.exp(1j * phase)
        
        # Create full object with support mask
        full_object = torch.zeros(self.num_pixels_object, dtype=torch.complex64, device=self.device)
        start = (self.num_pixels_object[0] - support_pixels) // 2
        end = start + support_pixels
        mask_central = self.support_mask[start:end, start:end]
        full_object[start:end, start:end] = object_field * mask_central
        
        # Set self.object for consistency with original method
        self.object.data = full_object
        return full_object

    def shift_diffuser(self, diffuser, shift):
        """Shift the diffuser field using the Fourier shift theorem."""
        dy, dx = shift
        vy = torch.fft.fftfreq(
            diffuser.shape[0], d=self.pixel_size_object, device=self.device
        )
        vx = torch.fft.fftfreq(
            diffuser.shape[1], d=self.pixel_size_object, device=self.device
        )
        vv, uu = torch.meshgrid(vy, vx, indexing="ij")
        phase_shift = torch.exp(-2j * np.pi * (uu * dx + vv * dy))
        diffuser_ft = torch.fft.fftn(diffuser)
        return torch.fft.ifftn(diffuser_ft * phase_shift)
    
    
    
    
    
    def Fres_propagate_test(self, field, object_spatial_res, wavelength, z, pad_factor):
        
        """
        Use direct implementation of FFT to compute the Fresnel-propagated field of the input-variable >field<
        The coordinates to which the propagated/computed field is scaled are not identical to those of the input variable's, as these
        are given by x_ij, y_ij = N_ij, M_ij * 1 / (Pad_factor * N_indices * \delta x) * wavelength * z
        = N_ij, M_ij * 1 / (Pad_factor * Length) * wavelength * z
        """
        
        
        ny, nx = field.shape
        padded_field = nn.functional.pad(
            field, (0, nx * (pad_factor - 1), 0, ny * (pad_factor - 1))
        )
        
        
        padded_field_dim_y = padded_field.shape[0];
        padded_field_dim_x = padded_field.shape[1];
        
        #Fourier-coords:
        fy_Fresnel = torch.fft.fftfreq(padded_field_dim_y, object_spatial_res);
        fx_Fresnel = torch.fft.fftfreq(padded_field_dim_x, object_spatial_res);
        
        fyy_Fresnel, fxx_Fresnel = torch.meshgrid(fy_Fresnel, fx_Fresnel, indexing = "ij");
        
        #Image-plane/Fresnel-coords:
        y_Fresnel = wavelength * z * fy_Fresnel;
        x_Fresnel = wavelength * z * fx_Fresnel;
        
        yy_Fresnel, xx_Fresnel = torch.meshgrid(y_Fresnel, x_Fresnel, indexing = "ij");
        
        #Object coords:
        y_Fresnel_object = object_spatial_res * padded_field_dim_y * torch.fft.fftfreq(padded_field_dim_y, 1);
        x_Fresnel_object = object_spatial_res * padded_field_dim_x * torch.fft.fftfreq(padded_field_dim_x, 1);
        
        Y_Fresnel_object, X_Fresnel_object = torch.meshgrid(y_Fresnel_object, x_Fresnel_object);
        
        #Fresnel quadratic phase
        Fresnel_quadratic_phase0 = torch.exp(1j * np.pi / (wavelength * z) * (X_Fresnel_object ** 2 + Y_Fresnel_object ** 2));
        
        Fresnel_quadratic_phase_wavelegnth_constr = (torch.sqrt((X_Fresnel_object ** 2 + Y_Fresnel_object ** 2)) <= 2 / wavelength);
        Fresnel_quadratic_phase_grid_res_constr = (np.pi / (wavelength * z) * 2 * torch.sqrt(X_Fresnel_object ** 2 + Y_Fresnel_object ** 2) <= 2/object_spatial_res);
        
        Fresnel_quadratic_mask_tot = Fresnel_quadratic_phase_grid_res_constr * Fresnel_quadratic_phase_wavelegnth_constr;
        
        Fresnel_quadratic_phase = Fresnel_quadratic_phase0 * Fresnel_quadratic_mask_tot;
        
        #In order to facilitate Fourier-transforming: Translating/rolling the image such that the original image's center coincides with that of the array's initial index
        padded_field_roll = torch.roll(padded_field, -int(ny/2), 0);
        padded_field_roll = torch.roll(padded_field_roll, -int(nx/2), 1);
        
        
        
        Fresnel_init_field = padded_field_roll * Fresnel_quadratic_phase;
        Fresnel_field0 = torch.fft.fft2(Fresnel_init_field);
        
        Fresnel_fringe_phase = torch.exp(1j * np.pi / (wavelength * z) * (xx_Fresnel ** 2 + yy_Fresnel ** 2));
        
        
        Fresnel_field0_phase = Fresnel_field0 * Fresnel_fringe_phase;
        
        Fresnel_field = torch.roll(Fresnel_field0_phase, int(ny/2), 0);
        Fresnel_field = torch.roll(Fresnel_field, int(nx/2), 1);
        
        index1_plot = 100;
        index2_plot = 900;

        
        return Fresnel_field


    def propagate(self, field):
        
        """Propagate the field using Fresnel diffraction with padding and cropping."""
        
        
        propagated = self.Fres_propagate_test(field, self.pixel_size_object, self.wavelength, self.prop_dist, self.pad_factor);
        
        print(self.pixel_size_object)
        print(propagated.shape)
        
        
        start_y = (field.shape[0] - self.crop_size) // 2
        start_x = (field.shape[1] - self.crop_size) // 2
        
        return propagated[
            start_y : start_y + self.crop_size, start_x : start_x + self.crop_size
        ]

    def simulate_stack(self):
        """
        Simulate a stack of diffraction patterns using the ground truth object.
        """
        # Initialize an empty list to collect patterns
        stack = []

        # Loop over each scanning position
        for shift in tqdm(self.scanning_positions):
            # Compute the forward model for this shift using the ground truth object
            # Use the current object data as ground truth for simulation
            intensity = self.compute_forward_model(
                shift, object_field=self.object.data
            )
            stack.append(intensity)

        # Stack all patterns into a single tensor
        self.diffraction_patterns = torch.stack(stack)
        return self.diffraction_patterns

    def compute_forward_model(self, shift, object_field=None):
        """Compute a single predicted diffraction pattern for a given shift."""
        if object_field is None:
            object_field = self.object

        # Shift the diffuser
        shifted_diffuser = self.shift_diffuser(self.diffuser, shift)

        # Crop shifted diffuser to match self.object size
        start_y = (shifted_diffuser.shape[0] - self.num_pixels_object[0]) // 2
        start_x = (shifted_diffuser.shape[1] - self.num_pixels_object[1]) // 2
        cropped_diffuser = shifted_diffuser[
            start_y : start_y + self.num_pixels_object[0],
            start_x : start_x + self.num_pixels_object[1],
        ]

        # Multiply with self.object
        interacted_field =  cropped_diffuser * object_field
        
        #plt.imshow(torch.abs(object_field).detach());plt.show()

        # Propagate and compute intensity
        propagated = self.propagate(interacted_field)
        return torch.abs(propagated) ** 2

    def optimize(
        self,
        num_iterations=1000,
        lr=0.01,
        lr_decay_factor=1.0,
        optimize_object=True,
        optimize_diffuser=False,
        shuffle_order=True,
        reg_amplitude=0,
        reg_binary=0,
        reg_support=0,
        reg_speckle_NA=0,
    ):
        # Renormalization step
        if self.total_photon_number is not None and optimize_diffuser:
            current_photon_number = torch.sum(torch.abs(self.diffuser) ** 2)
            self.diffuser = self.diffuser * torch.sqrt(
                self.total_photon_number / current_photon_number
            )
            print(
                f"Renormalized diffuser photon number: {torch.sum(torch.abs(self.diffuser)**2).item():.0f}"
            )

        params_to_optimize = []
        if optimize_object:
            params_to_optimize.append(self.object)
        if optimize_diffuser:
            self.diffuser = nn.Parameter(self.diffuser)
            params_to_optimize.append(self.diffuser)

        optimizer = optim.Adam(params_to_optimize, lr=lr)

        scheduler = None
        if 0.0 < lr_decay_factor < 1.0:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_factor)
            print(f"Using ExponentialLR scheduler with gamma={lr_decay_factor}")

        # Initialize history lists for total loss and individual terms
        loss_history = []
        data_loss_history = []
        binary_reg_history = []
        amplitude_reg_history = []
        support_reg_history = []
        speckle_reg_history = []

        # Progress bar with loss
        pbar = tqdm(range(num_iterations), desc="Optimization Progress")
        for iteration in pbar:
            total_loss = 0.0
            total_data_loss = 0.0
            total_binary_reg = 0.0
            total_amplitude_reg = 0.0
            total_support_reg = 0.0
            total_speckle_reg = 0.0

            num_positions = len(self.scanning_positions)
            if shuffle_order:
                # Generate shuffled indices for this epoch
                indices = torch.randperm(num_positions, device=self.device)
            else:
                # Generate ordered indices (0, 1, 2, ...)
                indices = torch.arange(num_positions, device=self.device)

            for idx in indices:
                i = idx.item() # Get the original integer index
                shift = self.scanning_positions[i] # Get shift using the index

                optimizer.zero_grad()
                predicted_intensity = self.compute_forward_model(shift)
                # Data fidelity loss (MSE on square roots)
                data_loss = nn.functional.mse_loss(
                    torch.sqrt(predicted_intensity) - torch.sqrt(self.diffraction_patterns[i]),
                    torch.zeros_like(predicted_intensity) ,
                ) + 0 * 1000e-6 * ((torch.sum(torch.abs(self.object - torch.roll(self.object, 1, 0)))
                     + torch.sum(torch.abs(self.object - torch.roll(self.object, 1, 1))))
                     + 0.5 * (torch.sum(torch.abs(self.object - torch.roll(self.object, 2, 0)))
                          + torch.sum(torch.abs(self.object - torch.roll(self.object, 2, 1))) + torch.sum(torch.abs(self.object - torch.roll(torch.roll(self.object, 2, 1), 2, 0))))
                     + 0.3 * (torch.sum(torch.abs(self.object - torch.roll(self.object, 3, 0)))
                          + torch.sum(torch.abs(self.object - torch.roll(self.object, 3, 1))))
                     )
                        
                        
                loss = data_loss

                # Regularization terms
                if optimize_object and reg_binary > 0:
                    binary_reg = reg_binary * torch.mean(
                        torch.abs(torch.abs(self.object) * (1 - torch.abs(self.object)))
                    )
                    loss += binary_reg / len(self.scanning_positions)
                    total_binary_reg += binary_reg.item() / len(self.scanning_positions)

                if optimize_diffuser and reg_amplitude > 0:
                    amplitude_reg = reg_amplitude * torch.mean(torch.abs(self.diffuser))
                    loss += amplitude_reg / len(self.scanning_positions)
                    total_amplitude_reg += amplitude_reg.item() / len(
                        self.scanning_positions
                    )

                if optimize_object and reg_support > 0:
                    support_reg = reg_support * torch.mean(
                        ((1 - self.support_mask) * torch.abs(self.object))
                    )
                    loss += support_reg / len(self.scanning_positions)
                    total_support_reg += support_reg.item() / len(
                        self.scanning_positions
                    )

                if optimize_diffuser and reg_speckle_NA > 0:
                    ft_diffuser = torch.fft.fftn(self.diffuser)
                    speckle_reg = reg_speckle_NA * torch.mean(
                        torch.abs(ft_diffuser * (1 - self.diffuser_mask)) ** 2
                    )
                    loss += speckle_reg / len(self.scanning_positions)
                    total_speckle_reg += speckle_reg.item() / len(
                        self.scanning_positions
                    )

                total_loss += loss.item()
                total_data_loss += data_loss.item()
                loss.backward()
                optimizer.step()

            # Compute averages over all scan positions
            avg_loss = total_loss / len(self.scanning_positions)
            avg_data_loss = total_data_loss / len(self.scanning_positions)
            avg_binary_reg = total_binary_reg  # Already averaged per position
            avg_amplitude_reg = total_amplitude_reg
            avg_support_reg = total_support_reg
            avg_speckle_reg = total_speckle_reg

            # Store in history lists
            loss_history.append(avg_loss)
            data_loss_history.append(avg_data_loss)
            binary_reg_history.append(avg_binary_reg)
            amplitude_reg_history.append(avg_amplitude_reg)
            support_reg_history.append(avg_support_reg)
            speckle_reg_history.append(avg_speckle_reg)

            if scheduler is not None:
                 scheduler.step()


            #self.diffuser = self.diffuser * 1/torch.abs(self.diffuser);

            pbar.set_description(f"Optimization Progress (Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.4f})")
            
            
            model.plot_complex(self.object, "Reconstructed Object")
            model.plot_complex(self.diffuser, "Reconstructed Diffuser")
            plt.imshow(torch.abs(self.object).detach(), cmap = 'hot');plt.show()

        # Plot all histories
        self.plot_loss_history(
            loss_history,
            data_loss_history,
            binary_reg_history,
            amplitude_reg_history,
            support_reg_history,
            speckle_reg_history,
        )

    def plot_loss_history(
        self,
        loss_history,
        data_loss_history,
        binary_reg_history,
        amplitude_reg_history,
        support_reg_history,
        speckle_reg_history,
    ):
        """
        Plot the total loss and individual regularization terms over iterations using a semilog plot.

        Args:
            loss_history (list): Total loss per iteration.
            data_loss_history (list): Data fidelity loss per iteration.
            binary_reg_history (list): Binary regularization term per iteration.
            amplitude_reg_history (list): Amplitude regularization term per iteration.
            support_reg_history (list): Support regularization term per iteration.
            speckle_reg_history (list): Speckle NA regularization term per iteration.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        iterations = range(len(loss_history))

        # Plot total loss
        plt.semilogy(
            iterations, loss_history, label="Total Loss", color="black", linewidth=2
        )

        # Plot data loss
        plt.semilogy(iterations, data_loss_history, label="Data Loss", color="blue")

        # Plot regularization terms if they are non-zero
        if any(binary_reg_history):
            plt.semilogy(
                iterations, binary_reg_history, label="Binary Reg", color="green"
            )
        if any(amplitude_reg_history):
            plt.semilogy(
                iterations, amplitude_reg_history, label="Amplitude Reg", color="orange"
            )
        if any(support_reg_history):
            plt.semilogy(
                iterations, support_reg_history, label="Support Reg", color="red"
            )
        if any(speckle_reg_history):
            plt.semilogy(
                iterations, speckle_reg_history, label="Speckle NA Reg", color="purple"
            )

        plt.xlabel("Iteration")
        plt.ylabel("Loss (log scale)")
        plt.title("Optimization Loss and Regularization History")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_complex(field, title=""):
        """Plot amplitude and phase of a complex field."""
        field = field.detach()
        amplitude = torch.abs(field).cpu().numpy()
        phase = torch.angle(field).cpu().numpy()
        amplitude = (amplitude - amplitude.min()) / (
            amplitude.max() - amplitude.min() + 1e-10
        )
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
        ax.set_facecolor("white")
        norm = plt.Normalize(-np.pi, np.pi)
        cmap = plt.get_cmap("hsv")
        phase_colors = cmap(norm(phase))
        phase_rgb = phase_colors[..., :3]
        colored_field = phase_rgb * amplitude[..., np.newaxis]
        ax.imshow(colored_field)
        plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Phase (radians)"
        )
        ax.set_title(title, color="black")
        plt.show()

    @staticmethod
    def plot_complex_subplots(field, title=""):
        """
        Plot amplitude and phase of a complex field in two subplots with colorbars.
        Amplitude colorbar is on the left, phase colorbar is on the right.

        Args:
            field (torch.Tensor): Complex field tensor.
            title (str): Overall title for the figure.
        """
        field = field.detach()
        amplitude = torch.abs(field).cpu().numpy()
        phase = torch.angle(field).cpu().numpy()

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

        # Amplitude subplot (left)
        ax1.set_facecolor("white")
        im1 = ax1.imshow(
            amplitude, cmap="gray", vmin=amplitude.min(), vmax=amplitude.max()
        )
        ax1.set_title("Amplitude", color="black")
        ax1.axis("off")
        cbar1 = plt.colorbar(
            im1, ax=ax1, orientation="vertical", pad=0.05, label="Amplitude"
        )
        cbar1.ax.yaxis.set_label_position(
            "left"
        )  # Move amplitude label to left side of colorbar
        cbar1.ax.yaxis.set_ticks_position("left")  # Move ticks to left side

        # Phase subplot (right)
        ax2.set_facecolor("white")
        im2 = ax2.imshow(phase, cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ax2.set_title("Phase", color="black")
        ax2.axis("off")
        plt.colorbar(
            im2, ax=ax2, orientation="vertical", pad=0.05, label="Phase (radians)"
        )

        # Overall figure title
        fig.suptitle(title, fontsize=16, color="black", y=1.05)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()




def sel_points_Fermat_spiral(init_inds, scanning_positions, dist_des, dist_des_tol_frac):
    
    """
    Function for generating a set of scanning-positions indices s.t. these are evenly spaced on the spiral
    dist_des : reference distance for dist. between scanning-positions
    dist_des_tol_frac : relative tolerance range used as creterium for adding a new index to the array of selected indices
    """

    init_scan_pos_y0 = scanning_positions[init_inds, 0];
    init_scan_pos_x0 = scanning_positions[init_inds, 1];

    init_scan_pos_y = np.copy(init_scan_pos_y0);
    init_scan_pos_x = np.copy(init_scan_pos_x0);

    dist_des_tol = dist_des_tol_frac * dist_des;

    for i in range(0, scanning_positions.shape[0]):
    
        if not(i in init_inds):
            scan_pos_x_i = scanning_positions[i,1];
            scan_pos_y_i = scanning_positions[i,0];
            print(i)
            dist_arr_i = np.sqrt((init_scan_pos_x - scan_pos_x_i) ** 2 + (init_scan_pos_y - scan_pos_y_i) ** 2);
            abs_dist_diff_i = np.abs(dist_arr_i - dist_des);
            
            inds_sel_i = np.where(abs_dist_diff_i <= 0.5 * dist_des_tol)[0];
            
            inds_dense =  np.where(dist_arr_i < dist_des - 0.5 * dist_des_tol)[0];
            
            #print(inds_dense)
            #print(inds_sel_i)
            #print(init_inds)
            
            if (len(inds_sel_i) > 0) and (len(inds_dense) == 0):
                inds_sel_ind1 = inds_sel_i[0];
                init_inds = np.append(init_inds, i);
                init_scan_pos_y = np.append(init_scan_pos_y, scanning_positions[i, 0]);
                init_scan_pos_x = np.append(init_scan_pos_x, scanning_positions[i, 1]);
            
            #print(abs_dist_diff_i)
                print('a')
                
    init_inds = np.unique(init_inds);
    return init_inds




#%
if __name__ == "__main__":

    params = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "wavelength": exp_wavelength,
        "total_photon_number": 1e6,  # rescales the diffuser amplitudes
        "prop_dist": exp_z_dist,
        "pixel_size_object": sim_obj_pix_size, #(0.2e-6),
        "num_pixels_object": sim_N_pix_obj,
        "M": 44.44,
        "speckle_NA": 0.6,
        "support_size": 448 * sim_obj_pix_size, #1 * 1*150e-6,
        "pad_factor": sim_pad_factor,
        "crop_size": sim_N_crop,
    }

    # Choose scanning pattern
    use_exp_scan = True  # Set to False to revert to linear scan
    if use_exp_scan:
        # Load scanning positions from Experiment 5 (Fermat spiral)
#        experiment_data = np.load(
#            os.path.join("data", "my_line_scan_20250311_152104.npz"),
#              allow_pickle=True
#        )
        
        #experiment_data = np.load(os.path.join("C:\\Users\\janbr\\Downloads\\Speckle_imaging_Fresh_Organoid_1_05082025", "organoid_scan_20250805_155641.npz"),
        #      allow_pickle=True)
        
        #data test Siemens star
        experiment_data = np.load("C:/Users/lotte/OneDrive/Documenten/Universiteit/BSc/Scriptie/Reconstructie/SiemensStarTest_20251001_125533.npz",allow_pickle=True)

        scanning_positions = 1 * experiment_data["scan_positions"][:,:] * 1e-6
        
        

        scanning_positions[:,0] = -scanning_positions[:,0];
        #scanning_positions[:,1] = -scanning_positions[:,1];
        
        """
        Below: selecting a subset of indices of scanning-positions lying at a certain distance w.r.t. each other
        Set variable dist_des_tol_frac to any number > 2 to select all indices / scanning-positions
        
        Set dist_des_tol_frac to a number > 2 to select all indices on scannin-pos. spiral
        
        """
        
        #Set dist_des_tol_frac to a number > 2 to select all indices on spiral
        dist_des = 0.0005;#0.0005;#0.0005;#0.0005;#0.0006;#0.00035;
        dist_des_tol_frac = 2.1;#1.2886#2.1;#2.1;#0.3;#1.2886;#0.8;#0.5;#0.3;#1.;#0.3;

        init_inds = np.array([0]);
        init_inds = sel_points_Fermat_spiral(init_inds, scanning_positions, dist_des, dist_des_tol_frac);#np.arange(1,401,2);#

        plt.figure(figsize = (5,5));plt.title('N-positions = ' + str(len(init_inds)))
        plt.scatter(scanning_positions[init_inds, 1], scanning_positions[init_inds, 0], marker = 'x', color = 'blue');plt.show()
        
        
        scanning_positions = scanning_positions[init_inds,:];
        
        
        if len(scanning_positions.shape) == 1:
            scanning_positions *= 1e3  # necessary for linear coords
            scanning_positions = np.stack(
                (np.zeros_like(scanning_positions), scanning_positions), axis=1
            )
    else:
        # Original linear scan along x-axis (commented out but preserved)
        scan_1D = np.linspace(0, 10e-3, 400)
        scanning_positions = np.stack((np.zeros_like(scan_1D), scan_1D), axis=1)

    speckle_data = {
        "scanning_positions": scanning_positions,
    }

    model = Correlated_Speckle_Imaging(params, speckle_data)

    # %% initialize ground truth object and diffuser
    ground_truth_object = model.generate_phase_object()
    model.plot_complex(model.object, "Ground Truth Object")
    model.plot_complex(model.diffuser, "Ground Truth Diffuser")

#%% OPTIONAL : Use an recorded camera-image as initial guess for the object 

from PIL import Image
#exp_data_imagesground_truth_object = np.load(os.path.join("C:\\Users\\janbr\\Downloads\\250404_speckle_illumination_imaging_round3", "Image__2025-04-04__14-37-46_Siemens_star_Use_as_Ground_Truth_JPG.jpg"))

#exp_data_imagesground_truth_object = Image.open(os.path.join("C:\\Users\\janbr\\Downloads\\250404_speckle_illumination_imaging_round3", "Image__2025-04-04__14-37-46_Siemens_star_Use_as_Ground_Truth_JPG.jpg"))
#exp_data_imagesground_truth_object_speckled = Image.open(os.path.join("C:\\Users\\janbr\\Downloads\\250404_speckle_illumination_imaging_round3", "Image__2025-04-04__14-38-16_Siemens_star_Ground_Truth_equiv_but_speckled_JPG.jpg"))

#ground_truth_object = Image.open(os.path.join("C:\\Users\\janbr\\Downloads\\Speckle_imaging_rounds_continued_from_25_04_2025", "Image__2025-04-25__09-39-24_Siemens_star_ground_truth_JPG.jpg"))
exp_data_imagesground_truth_object = Image.open(os.path.join("C:\\Users\\janbr\\Downloads\\Speckle_imaging_rounds_continued_from_25_04_2025", "Image__2025-04-25__09-39-24_Siemens_star_ground_truth_JPG.jpg"))

#exp_data_imagesground_truth_object = np.load(os.path.join("C:\\Users\\janbr\\Downloads\\250404_speckle_illumination_imaging_round3", "Image__2025-04-04__14-37-46_Siemens_star_Use_as_Ground_Truth_JPG.jpg"))
exp_data_imagesground_truth_object_speckled = Image.open(os.path.join("C:\\Users\\janbr\\Downloads\\Speckle_imaging_rounds_continued_from_25_04_2025", "Image__2025-04-25__09-37-33_Siemens_star_Ground_truth_but_speckled_JPG.jpg"))

exp_data_imagesground_truth_object = np.swapaxes(exp_data_imagesground_truth_object, 0, 1);
#exp_data_imagesground_truth_object_speckled = np.swapaxes(exp_data_imagesground_truth_object_speckled, 0, 1);


exp_data_imagesground_truth_object = np.flip(exp_data_imagesground_truth_object, 0);

exp_data_imagesground_truth_object = np.array(exp_data_imagesground_truth_object)[:]
exp_data_imagesground_truth_object_speckled = np.array(exp_data_imagesground_truth_object_speckled);

exp_data_imagesground_truth_object = rgb2gray(exp_data_imagesground_truth_object);
exp_data_imagesground_truth_object_speckled = rgb2gray(exp_data_imagesground_truth_object_speckled);

#exp_data_imagesground_truth_object = exp_data_imagesground_truth_object[200:200 + 2048,:][::2,::2];
#exp_data_imagesground_truth_object_speckled = exp_data_imagesground_truth_object_speckled[200:200 + 2048,:][::2,::2];

exp_data_imagesground_truth_object = exp_data_imagesground_truth_object[104:104 + 1024, 2:2 + 1024];
exp_data_imagesground_truth_object_speckled = exp_data_imagesground_truth_object_speckled[104:104 + 1024, 2:2 + 1024];

exp_data_imagesground_truth_object = np.roll(exp_data_imagesground_truth_object, -50, 1);
exp_data_imagesground_truth_object = np.roll(exp_data_imagesground_truth_object, 60, 0);

plt.imshow(exp_data_imagesground_truth_object, cmap = 'gray');plt.show()
plt.imshow(exp_data_imagesground_truth_object_speckled, cmap = 'gray');plt.show()


#%% OPTIONAL : Use an recorded camera-image as initial guess for the object 

from PIL import Image
#exp_data_imagesground_truth_object = np.load(os.path.join("C:\\Users\\janbr\\Downloads\\250404_speckle_illumination_imaging_round3", "Image__2025-04-04__14-37-46_Siemens_star_Use_as_Ground_Truth_JPG.jpg"))

#exp_data_imagesground_truth_object = Image.open(os.path.join("C:\\Users\\janbr\\Downloads\\250404_speckle_illumination_imaging_round3", "Image__2025-04-04__14-37-46_Siemens_star_Use_as_Ground_Truth_JPG.jpg"))
#exp_data_imagesground_truth_object_speckled = Image.open(os.path.join("C:\\Users\\janbr\\Downloads\\250404_speckle_illumination_imaging_round3", "Image__2025-04-04__14-38-16_Siemens_star_Ground_Truth_equiv_but_speckled_JPG.jpg"))


exp_data_imagesground_truth_object = Image.open(os.path.join("C:\\Users\\janbr\\Downloads\\Speckle_imaging_attempt_paper_sample_crescent_28_04_2025", "Image__2025-04-28__13-50-53_crescent_ground_truth_JPG.jpg"))
exp_data_imagesground_truth_object_speckled = Image.open(os.path.join("C:\\Users\\janbr\\Downloads\\Speckle_imaging_attempt_paper_sample_crescent_28_04_2025", "Image__2025-04-28__13-50-16_crescent_ground_truth_speckled_JPG.jpg"))


exp_data_imagesground_truth_object = np.array(exp_data_imagesground_truth_object)
exp_data_imagesground_truth_object_speckled = np.array(exp_data_imagesground_truth_object_speckled);

exp_data_imagesground_truth_object = rgb2gray(exp_data_imagesground_truth_object);
exp_data_imagesground_truth_object_speckled = rgb2gray(exp_data_imagesground_truth_object_speckled);

exp_data_imagesground_truth_object = np.swapaxes(exp_data_imagesground_truth_object, 0, 1);
exp_data_imagesground_truth_object = np.flip(exp_data_imagesground_truth_object, 0);

#exp_data_imagesground_truth_object = exp_data_imagesground_truth_object[:,200:200 + 2048][::2,::2];
#exp_data_imagesground_truth_object_speckled = exp_data_imagesground_truth_object_speckled[:,200:200 + 2048][::2,::2];


exp_data_imagesground_truth_object = exp_data_imagesground_truth_object[4:4 + 1024,104:104 + 1024];
exp_data_imagesground_truth_object_speckled = exp_data_imagesground_truth_object_speckled[4:4 + 1024,104:104 + 1024];

plt.imshow(exp_data_imagesground_truth_object, cmap = 'gray');plt.show()
plt.imshow(exp_data_imagesground_truth_object_speckled, cmap = 'gray');plt.show()



#%% Load the experimental data (camera images)

exp_data_images0 = experiment_data["images"];

#%% Permute & flip axes in order to match the coordinate-grids used in the simulation

exp_data_images = exp_data_images0[init_inds,:]
#exp_data_images = np.swapaxes(exp_data_images, 1, 2);

exp_data_images = np.flip(exp_data_images, 1);
exp_data_images = np.flip(exp_data_images, 2);




N_exp_images = exp_data_images.shape[0];
print(exp_data_images.shape)

#%% OPTIONAL: Section intended to remove eventual noise from the camera images

average_data_imgs = np.mean(exp_data_images, 0);

plt.imshow(average_data_imgs);plt.colorbar();plt.show()

for i in range(0, exp_data_images.shape[0]):
    exp_data_images[i,:,:] += -average_data_imgs;

abs_min_data_imgs = np.min(exp_data_images);

exp_data_images += -abs_min_data_imgs;

average_data_imgs_fin = np.mean(exp_data_images, 0);

plt.imshow(average_data_imgs_fin);plt.colorbar();plt.show()


#%% OPTIONAL: Section intended to remove eventual noise from the camera images

err_pixels = np.where(average_data_imgs >= 1000);

exp_data_images[:,err_pixels[0], err_pixels[1]] = 0;

    
for e1 in [-1,1]:
    for e2 in [-1,1]:
        exp_data_images[:,err_pixels[0], err_pixels[1]] += 1/4 * exp_data_images[:,err_pixels[0] + e1, err_pixels[1] + e2];

average_data_imgs_fin = np.mean(exp_data_images, 0);

plt.imshow(average_data_imgs_fin);plt.colorbar();plt.show()

#%% Section intended to rescale the camera-images to match the number of pixels of the cropped diffraction pattern

#% Generate a grid for camera-image in actual spatial coords.
cam_pixel_size_x, cam_pixel_size_y = cam_pix_size, cam_pix_size;


N_pixels_x = exp_data_images.shape[-1]
N_pixels_y = exp_data_images.shape[-2]

x_exp_pixels_arr = cam_pixel_size_x * np.arange(0, N_pixels_x);
y_exp_pixels_arr = cam_pixel_size_y * np.arange(0, N_pixels_y);

[X_exp_pixels_arr, Y_exp_pixels_arr] = np.meshgrid(x_exp_pixels_arr, y_exp_pixels_arr);


#% Generate a grid for the camera-image matching the specified cropped-size of the simulated diffraction pattern
exp_images_reshape_size = model.crop_size;

x_exp_pixels_res = np.linspace(0, x_exp_pixels_arr[-1], exp_images_reshape_size);
y_exp_pixels_res = np.linspace(0, y_exp_pixels_arr[-1], exp_images_reshape_size);

[X_exp_pixels_res, Y_exp_pixels_res] = np.meshgrid(x_exp_pixels_res, y_exp_pixels_res);

Arr_store_res_coords = np.array([X_exp_pixels_res, Y_exp_pixels_res]);
Arr_store_res_coords = np.swapaxes(Arr_store_res_coords, 0, 2);
#Arr_store_res_coords = np.swapaxes(Arr_store_res_coords, 0, 1);


#%Rescale the camera-image s.t. it matches grid of the cropped simulated diffraction pattern

exp_data_images_res = np.zeros((N_exp_images, exp_images_reshape_size, exp_images_reshape_size));

for i in range(0, N_exp_images):
    exp_data_img_i = np.squeeze(exp_data_images[i,:,:]);
    exp_data_img_i_intp = interpn((x_exp_pixels_arr, y_exp_pixels_arr), exp_data_img_i, Arr_store_res_coords)
    exp_data_images_res[i] = exp_data_img_i_intp;
    print(i)


#%
model.diffraction_patterns = torch.tensor(exp_data_images_res, dtype = torch.float32);

    #%% Simulate diffraction patterns
    #Niet runnen!
print("Simulating diffraction patterns...")
with torch.no_grad():
    model.simulate_stack()
print(f"Simulated stack shape: {model.diffraction_patterns.shape}")

    # visualize 2 first diffraction patterns
for i in range(2):
    model.plot_complex(model.diffraction_patterns[i])
        

#%% (P)reset object and diffuser (optionally with prior knowledge) for reconstruction
#model.object.data = 0.0 * torch.tensor(Obj_image_resized, dtype = torch.complex64) + 1.0 * torch.ones_like(
#    model.object, dtype=torch.complex64, device=model.device
#)


model.object.data = 0.0 * model.object + 1.0 * torch.ones_like(
    model.object, dtype=torch.complex64, device=model.device
)

model.diffuser = 0.0 * model.diffuser + 1.0 * torch.ones_like(
    model.diffuser, dtype=torch.complex64, device=model.device
)

# model.diffuser_seed = 123  # set to different speckle pattern
# model.total_photon_number = None
# model.initialize_diffuser()


# Manual check of initial loss before optimization
with torch.no_grad():
    pred = model.compute_forward_model(model.scanning_positions[0])
    loss = nn.functional.mse_loss(
        torch.sqrt(pred), torch.sqrt(model.diffraction_patterns[0])
    )
    print(f"Initial loss for first position: {loss.item():.16f}")

    #%% Optimize to reconstruct the object and diffuser
    print("Starting optimization...")
    model.optimize(
        num_iterations=25, 
        lr=0.02, #Set to ~0.02 for experimental reconstructions without prior knowledge, set to ~0.05 for reconstructing simulated diffraction patterns
#        lr_decay_factor=0.98,
        optimize_object=True,
        optimize_diffuser=True,
        shuffle_order=True,
        reg_amplitude=1e-3,
        # reg_binary=1e-2,  # not so useful I think
        reg_support=0.1,
        reg_speckle_NA=1e-0,
    )

    # Visualize reconstructed object and diffuser
    model.plot_complex(model.object, "Reconstructed Object")
    model.plot_complex(model.diffuser, "Reconstructed Diffuser")

    # visualize with amplitude colorbar
    model.plot_complex_subplots(model.object, "Reconstructed Object")
    model.plot_complex_subplots(model.diffuser, "Reconstructed Diffuser")

#%% Everything above this cell is written by J.B.M.Y. Heinisch

#
#  Pixel version finite difference one direction
# plaatje simuleren, daar ruis aan toevoegen, dan positie bepalen
def fisher_infox(shift, difference):
    shift0 = shift
    shift_left[0] = shift0[0] - difference[0]
    shift_left[1] = shift0[1] - difference[1]
    shift_right[0] = shift0[0] + difference[0]
    shift_right[1] = shift0[1] + difference[1]
    # Compute forward model for this position
    I0 = model.compute_forward_model(shift0)
    Ileft = model.compute_forward_model(shift_left)
    Iright = model.compute_forward_model(shift_right)
    # Finite difference to find slope
    dI_dx = torch.div((Iright - Ileft),2*difference[1])
    #Fisher information per pixel
    Fisher_information_tensor = torch.div( torch.mul(dI_dx, dI_dx), I0)
    Fisher_information = torch.sum(Fisher_information_tensor)
    print(Fisher_information)
    CRLB = 1 / Fisher_information.item()
    sigma = np.sqrt(CRLB)
    return Fisher_information.item(), dI_dx, sigma.item()

print(f" The standard deviation is geq {(fisher_infox([0,0],[0,100e-9])[2])*1e9} nm.")
# %%
def slopex(shift, difference):   
    shift0 = shift
    shift_left   == [shift[0], shift[1] - difference]
    shift_right = [shift[0],shift[1] + difference]
    # Compute forward model for this position
    Ileft = model.compute_forward_model(shift_left)
    Iright = model.compute_forward_model(shift_right)
    # Finite difference to find slope
    dI_dx = torch.div((Iright - Ileft),2*difference)
    return dI_dx

def slopey(shift, difference):   
    shift_down = [shift[0] - difference, shift[1]]
    shift_up = [shift[0] + difference,shift[1] ]
    # Compute forward model for this position
    Idown = model.compute_forward_model(shift_down)
    Iup = model.compute_forward_model(shift_up)
    # Finite difference to find slope
    dI_dy = torch.div((Iup - Idown),2*difference)
    return dI_dy

def fisher_info(shift, difference):
    shift0 = shift
    # print(difference)
    dI_dx = slopex(shift, difference)
    dI_dy = slopey(shift, difference)
    I0 = model.compute_forward_model(shift0)
    #Fisher information per pixel
    F00= torch.sum(torch.div( torch.mul(dI_dx, dI_dx), I0)).item()
    F01= torch.sum(torch.div( torch.mul(dI_dx, dI_dy), I0)).item()
    F11= torch.sum(torch.div( torch.mul(dI_dy, dI_dy), I0)).item()
    return [[F00, F01],[F01,F11]]

A = torch.tensor(fisher_info([0,0],5e-9))
print(A)
print(torch.linalg.inv(A))

# %% Turning
def slope(shift, difference):   
    shift_left  = np.subtract(shift,difference)
    shift_right = np.add(shift,difference)
    # Compute forward model for this position
    Ileft = model.compute_forward_model(shift_left)
    Iright = model.compute_forward_model(shift_right)
    norm = np.sqrt(difference[0]**2+difference[1]**2)
    # Finite difference to find slope
    dI_d1 = torch.div((Iright - Ileft),2*norm)
    return dI_d1

def fisher_info(shift, difference1,difference2):
    # print(difference)
    dI_d1 = slope(shift, difference1)
    dI_d2 = slope(shift, difference2)
    I0 = model.compute_forward_model(shift)
    #Fisher information per pixel
    F00= torch.sum(torch.div( torch.mul(dI_d1, dI_d1), I0)).item()
    F01= torch.sum(torch.div( torch.mul(dI_d1, dI_d2), I0)).item()
    F11= torch.sum(torch.div( torch.mul(dI_d2, dI_d2), I0)).item()
    return [[F00, F01],[F01,F11]]

A = torch.tensor(fisher_info([0,0],[0,5e-9],[5e-9,0]))
print(A) #Fisher info matrix
print(torch.linalg.inv(A)) #inverse of FI, needed for CRLB

#%% Angle test, the FI matrix depends on the orientation of the object
step = 5e-9
t = np.linspace(0, np.pi,num=30)
F10 = []
F00 = []
Fdif = []
for ti in t:
    F = fisher_info([0,0],[step * np.cos(ti),step *np.sin(ti)], [step* np.cos(ti-np.pi/2),step* np.sin(ti-np.pi/2)])
    F10.append(F[1][0])
    F00.append(F[0][0])
    Fdif.append(F[0][0]-F[1][0])

plt.plot(t, F10)
plt.plot(t, F00)
plt.plot(t,Fdif)
plt.xlabel("Angle")
plt.ylabel("Cross term")

# %% Stability check for finite difference method
diff = np.linspace(0.000001, 0.00002, num=100)

dI_dx_val1 = []
dI_dx_val2 = []
dI_dx_valdark = []
for d in diff:
    _, dI_dx = fisher_infox([0, 0], [0, d])
    dI_dx_val1.append(dI_dx[400][400].item())
    dI_dx_val2.append(dI_dx[320][450].item())
    dI_dx_valdark.append(dI_dx[200][410].item()) #dark pixel

plt.plot(diff, dI_dx_val1)
plt.plot(diff, dI_dx_val2)
plt.plot(diff, dI_dx_valdark)
plt.xlabel("Finite difference step size")
plt.ylabel("dI/dx")
plt.title("Derivative vs. step size")
plt.show()

# %% Check for dark/broken pixels
for i in range(int(i_min), int(i_max)):
    for j in range(int(j_min), int(j_max)):
        I_j = I[i,j]
        if I_j.item() < 100:
            print(f"Pixel ({i},{j}) has very low intensity: {I_j.item()}")  
# %%
