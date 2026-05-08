clc; clear all;
tic

% Load and preprocess the star image
load star_input_data.mat

phase_star = unwrap_star_background_zero;

clear unwrap_star_background_zero

figure; imagesc(phase_star); title('Unwrapped Phase Map Star'); colorbar;



% Define parameters
[Nx Ny] = size(phase_star);
lambda = 0.532;              % Wavelength in microns
RI = 1.5148255; %https://www.benchmarktech.com/quantitativephasemicroscop/

d_true = phase_microlens.*(lambda / (2*pi*(RI-1)));
figure; imagesc(d_true); axis image;  title('True thickness map'); colorbar;

n_true = phase_microlens.*(lambda/ (2*pi*0.350));
figure; imagesc(n_true); axis image;  title('True RI map'); colorbar;

%%  Convert ranges to GPU arrays

% Preallocate arrays for storing results
%n1_optimal = gpuArray.zeros(N, N);  % Preallocate a GPU array of zeros for n1_optimal
%n2_optimal = gpuArray.zeros(N, N);  % Preallocate a GPU array of zeros for n2_optimal
%d_optimal = gpuArray.zeros(N, N);   % Preallocate a GPU array of zeros for d_optimal

n1_hat = gpuArray(1.3:0.0001:1.8);
n2_hat = gpuArray(0.00100:0.00001:0.00500);
d_hat =  gpuArray(0:0.01:2);

% Create Grid
[N1, N2, D] = ndgrid(n1_hat, n2_hat, d_hat);
phase_hat = 2 * pi  .* (N1 + N2 ./ lambda.^2 - 1) .* D./ lambda; % Estimated phase calculation on the GPU


for i = 1:Nx
    for j = 1:Ny
        reconstructed_phase_1d = phase_star(i,j); % Extract true phase value for the current pixel
        Jcost = (reconstructed_phase_1d - phase_hat).^2;  % Objective function (Jcost) calculation on the GPU
        Minimum_Value_gpu = min(Jcost, [],'all'); % Find minimum Jcost/error for each d_hat on the GPU
        Minimum_Value = gather(Minimum_Value_gpu);  % Bring minimum value back to CPU
        [n1_loc, n2_loc, d_loc] = ind2sub(size(Jcost), find(Jcost == Minimum_Value, 1, 'first'));

        % Store optimal values for the current pixel (gather to bring data to CPU)
        % Assign temporary arrays back to the main array
        n1_optimal(i, j) = gather(n1_hat(n1_loc));
        n2_optimal(i, j)  =gather(n2_hat(n2_loc));
        d_optimal(i, j)  =  gather(d_hat(d_loc));

    end
end
%% Display results

ns_lambda_optimal = n1_optimal + n2_optimal ./ lambda.^2;

figure;imagesc(ns_lambda_optimal);axis image; title('Estimated RI')


figure;imagesc(d_optimal);axis image; title('Estimated thickness')


toc