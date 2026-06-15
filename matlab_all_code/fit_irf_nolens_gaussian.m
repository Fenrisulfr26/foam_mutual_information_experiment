function fit_irf_nolens_gaussian()
% FIT_IRF_NOLENS_GAUSSIAN Fit 1D Gaussian to 32x32xN histograms for each nolens file
% in the IRF folder, and save/output the peak positions and widths.

% Get the current script folder as base
PROJECT_ROOT = fileparts(mfilename('fullpath'));
irf_dir = fullfile(PROJECT_ROOT, 'IRF');

if ~isfolder(irf_dir)
    error('IRF directory not found at: %s', irf_dir);
end

files = dir(fullfile(irf_dir, '*nolens*.mat'));
if isempty(files)
    fprintf('No files matching *nolens*.mat found in: %s\n', irf_dir);
    return;
end

fprintf('Found %d files to process.\n\n', length(files));

for f = 1:length(files)
    filename = files(f).name;
    filepath = fullfile(irf_dir, filename);
    fprintf('Processing file %d/%d: %s...\n', f, length(files), filename);

    % Load the data
    data = load(filepath);
    if isfield(data, 'hist')
        H = data.hist;
    elseif isfield(data, 'hist_raw')
        H = data.hist_raw;
    else
        warning('No hist or hist_raw variable found in %s', filename);
        continue;
    end

    [ny, nx, nBins] = size(H);
    fprintf('  Data dimensions: %d x %d x %d\n', ny, nx, nBins);

    % Preallocate variables for results
    peak_positions = zeros(ny, nx);
    peak_widths = zeros(ny, nx);
    amplitudes = zeros(ny, nx);
    baselines = zeros(ny, nx);
    fit_errors = zeros(ny, nx);

    options = optimset('Display', 'off', 'MaxIter', 500, 'MaxFunEvals', 1000);

    tic;
    for y = 1:ny
        for x = 1:nx
            h_curve = squeeze(H(y, x, :));

            % If the signal is too weak, skip fitting
            if sum(h_curve) < 10 || max(h_curve) < 3
                peak_positions(y, x) = NaN;
                peak_widths(y, x) = NaN;
                amplitudes(y, x) = NaN;
                baselines(y, x) = NaN;
                fit_errors(y, x) = NaN;
                continue;
            end

            [max_val, max_idx] = max(h_curve);

            % Define fit window around peak to isolate the main peak
            window_half_width = 15; % 31 bins total
            w_start = max(1, max_idx - window_half_width);
            w_end = min(nBins, max_idx + window_half_width);

            t_fit = (w_start:w_end)';
            y_fit = double(h_curve(w_start:w_end));

            % Initial guesses
            offset0 = min(y_fit);
            amp0 = max_val - offset0;
            if amp0 <= 0, amp0 = 1; end
            mu0 = double(max_idx);
            sigma0 = 2.0; % typical width of IRF peak is narrow

            p0 = [amp0, mu0, sigma0, offset0];

            % Objective function for the windowed data
            objective = @(p) ...
                (p(1) <= 0 || p(3) <= 0.1 || p(3) > 30 || p(2) < w_start || p(2) > w_end) * 1e12 + ...
                sum((y_fit - (p(1) * exp(-0.5 * ((t_fit - p(2)) ./ p(3)).^2) + p(4))).^2);

            try
                p_fit = fminsearch(objective, p0, options);

                % Check if fit is valid (not penalized)
                if objective(p_fit) > 1e10
                    % Fallback to max_idx and default sigma if fit failed
                    peak_positions(y, x) = mu0;
                    peak_widths(y, x) = sigma0;
                    amplitudes(y, x) = amp0;
                    baselines(y, x) = offset0;
                    fit_errors(y, x) = -1;
                else
                    amplitudes(y, x) = p_fit(1);
                    peak_positions(y, x) = p_fit(2);
                    peak_widths(y, x) = p_fit(3);
                    baselines(y, x) = p_fit(4);

                    % Calculate fitting error (RMSE)
                    fit_curve = p_fit(1) * exp(-0.5 * ((t_fit - p_fit(2)) ./ p_fit(3)).^2) + p_fit(4);
                    rmse = sqrt(mean((y_fit - fit_curve).^2));
                    fit_errors(y, x) = rmse;
                end
            catch
                % Fallback
                peak_positions(y, x) = mu0;
                peak_widths(y, x) = sigma0;
                amplitudes(y, x) = amp0;
                baselines(y, x) = offset0;
                fit_errors(y, x) = -2;
            end
        end
    end
    elapsed = toc;
    fprintf('  Fitting completed in %.2f seconds.\n', elapsed);

    % Save results
    [~, base_name, ~] = fileparts(filepath);
    out_filename = [base_name '_fit_results.mat'];
    out_filepath = fullfile(irf_dir, out_filename);

    save(out_filepath, 'peak_positions', 'peak_widths', 'amplitudes', 'baselines', 'fit_errors');
    fprintf('  Saved fit results to: %s\n', out_filename);

    % Print summary stats (ignoring NaNs)
    valid_pos = peak_positions(~isnan(peak_positions));
    valid_wid = peak_widths(~isnan(peak_widths));

    if ~isempty(valid_pos)
        fprintf('  Peak position (mu): Mean = %.3f, Std = %.3f, Min = %.3f, Max = %.3f (bins)\n', ...
            mean(valid_pos), std(valid_pos), min(valid_pos), max(valid_pos));
        fprintf('  Peak width (sigma): Mean = %.3f, Std = %.3f, Min = %.3f, Max = %.3f (bins)\n', ...
            mean(valid_wid), std(valid_wid), min(valid_wid), max(valid_wid));
        fprintf('  Peak width (FWHM): Mean = %.3f (bins)\n', mean(valid_wid) * 2.3548);
    else
        fprintf('  No valid peaks found.\n');
    end
    fprintf('\n');
end
end
