function analyze_results()
    % Get the current script folder as base
    PROJECT_ROOT = fileparts(mfilename('fullpath'));
    irf_dir = PROJECT_ROOT;
    
    files = {
        'IRF_noLens_10avg_20260612_2210_fit_results.mat', ...
        'IRF_noLens_20260612_2210_fit_results.mat', ...
        'IRF_nolens_paper_10_avg_20261712_220615_fit_results.mat', ...
        'IRF_nolens_paper_20261712_220615_fit_results.mat'
    };
    
    report_file = fullfile(irf_dir, 'analysis_report.txt');
    fid = fopen(report_file, 'w');
    if fid == -1
        error('Cannot open report file for writing');
    end
    
    fprintf(fid, '==================================================\n');
    fprintf(fid, 'SPAD 32x32 IRF Gaussian Fit Analysis Report\n');
    fprintf(fid, 'Report Generated: %s\n', datestr(now));
    fprintf(fid, '==================================================\n\n');
    
    for i = 1:length(files)
        filename = files{i};
        filepath = fullfile(irf_dir, filename);
        
        fprintf(fid, '--------------------------------------------------\n');
        fprintf(fid, 'File: %s\n', filename);
        fprintf(fid, '--------------------------------------------------\n');
        
        if ~exist(filepath, 'file')
            fprintf(fid, 'Error: File does not exist.\n\n');
            continue;
        end
        
        % Load variables
        data = load(filepath);
        
        if ~isfield(data, 'peak_positions')
            fprintf(fid, 'Error: peak_positions not found in file.\n\n');
            continue;
        end
        
        pos = data.peak_positions;
        wid = data.peak_widths;
        amp = data.amplitudes;
        base = data.baselines;
        err = data.fit_errors;
        
        % Filter out NaNs and invalid fits
        valid_idx = ~isnan(pos) & ~isnan(wid);
        num_valid = sum(valid_idx(:));
        total_pixels = numel(pos);
        
        fprintf(fid, '  Total Pixels: %d, Successfully Fitted: %d (%.2f%%)\n\n', ...
            total_pixels, num_valid, (num_valid / total_pixels) * 100);
        
        if num_valid > 0
            v_pos = pos(valid_idx);
            v_wid = wid(valid_idx);
            v_amp = amp(valid_idx);
            v_base = base(valid_idx);
            v_err = err(valid_idx);
            
            % FWHM in bins
            v_fwhm = v_wid * 2.3548;
            
            % Peak positions (Timing skew across SPAD array)
            fprintf(fid, '  1. Peak Positions (mu, in bins):\n');
            fprintf(fid, '     Mean   = %.4f bins\n', mean(v_pos));
            fprintf(fid, '     Median = %.4f bins\n', median(v_pos));
            fprintf(fid, '     Std    = %.4f bins  <-- (Timing Skew / Jitter across array)\n', std(v_pos));
            fprintf(fid, '     Min    = %.4f bins\n', min(v_pos));
            fprintf(fid, '     Max    = %.4f bins\n', max(v_pos));
            fprintf(fid, '     Peak-to-Peak Skew = %.4f bins\n\n', max(v_pos) - min(v_pos));
            
            % Peak widths (Sigma, in bins)
            fprintf(fid, '  2. Peak Widths (sigma, in bins):\n');
            fprintf(fid, '     Mean   = %.4f bins\n', mean(v_wid));
            fprintf(fid, '     Median = %.4f bins\n', median(v_wid));
            fprintf(fid, '     Std    = %.4f bins\n', std(v_wid));
            fprintf(fid, '     Min    = %.4f bins\n', min(v_wid));
            fprintf(fid, '     Max    = %.4f bins\n\n', max(v_wid));
            
            % FWHM (Full Width at Half Maximum)
            fprintf(fid, '  3. FWHM (Full Width at Half Maximum, in bins):\n');
            fprintf(fid, '     Mean   = %.4f bins\n', mean(v_fwhm));
            fprintf(fid, '     Median = %.4f bins\n', median(v_fwhm));
            fprintf(fid, '     Std    = %.4f bins\n', std(v_fwhm));
            fprintf(fid, '     Min    = %.4f bins\n', min(v_fwhm));
            fprintf(fid, '     Max    = %.4f bins\n\n', max(v_fwhm));
            
            % Amplitudes (Signal strength)
            fprintf(fid, '  4. Amplitudes (Peak Photon Count, counts):\n');
            fprintf(fid, '     Mean   = %.2f\n', mean(v_amp));
            fprintf(fid, '     Median = %.2f\n', median(v_amp));
            fprintf(fid, '     Std    = %.2f\n', std(v_amp));
            fprintf(fid, '     Min    = %.2f\n', min(v_amp));
            fprintf(fid, '     Max    = %.2f\n\n', max(v_amp));
            
            % Baselines (Noise floor)
            fprintf(fid, '  5. Baselines (Background / Dark count floor):\n');
            fprintf(fid, '     Mean   = %.2f\n', mean(v_base));
            fprintf(fid, '     Median = %.2f\n', median(v_base));
            fprintf(fid, '     Std    = %.2f\n', std(v_base));
            fprintf(fid, '     Min    = %.2f\n', min(v_base));
            fprintf(fid, '     Max    = %.2f\n\n', max(v_base));
            
            % Fit error (RMSE)
            fprintf(fid, '  6. Fit Errors (RMSE, counts):\n');
            fprintf(fid, '     Mean   = %.4f\n', mean(v_err));
            fprintf(fid, '     Median = %.4f\n', median(v_err));
            fprintf(fid, '     Std    = %.4f\n', std(v_err));
        else
            fprintf(fid, '  No valid fitted pixels.\n\n');
        end
        fprintf(fid, '\n');
    end
    
    fclose(fid);
    fprintf('Analysis report written to: %s\n', report_file);
end
