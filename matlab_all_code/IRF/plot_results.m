function plot_results()
    % Get the current script folder as base
    PROJECT_ROOT = fileparts(mfilename('fullpath'));
    irf_dir = PROJECT_ROOT;
    
    file1 = fullfile(irf_dir, 'IRF_noLens_10avg_20260612_2210_fit_results.mat');
    file2 = fullfile(irf_dir, 'IRF_nolens_paper_10_avg_20261712_220615_fit_results.mat');
    
    if ~exist(file1, 'file') || ~exist(file2, 'file')
        error('Required fit results files do not exist!');
    end
    
    d1 = load(file1);
    d2 = load(file2);
    
    % Set up figure (invisible)
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 1000]);
    
    colormap_choice = 'viridis'; % MATLAB default parula is safer, let's use parula or jet
    
    % --- Row 1: Peak Positions (Timing Skew) ---
    subplot(3, 2, 1);
    imagesc(d1.peak_positions);
    title('20260612: Peak Positions (\mu, bins)');
    colorbar;
    axis image;
    xlabel('X pixel'); ylabel('Y pixel');
    
    subplot(3, 2, 2);
    imagesc(d2.peak_positions);
    title('Paper: Peak Positions (\mu, bins)');
    colorbar;
    axis image;
    xlabel('X pixel'); ylabel('Y pixel');
    
    % --- Row 2: FWHM ---
    subplot(3, 2, 3);
    imagesc(d1.peak_widths * 2.3548);
    title('20260612: FWHM (bins)');
    colorbar;
    axis image;
    xlabel('X pixel'); ylabel('Y pixel');
    
    subplot(3, 2, 4);
    imagesc(d2.peak_widths * 2.3548);
    title('Paper: FWHM (bins)');
    colorbar;
    axis image;
    xlabel('X pixel'); ylabel('Y pixel');
    
    % --- Row 3: Amplitudes ---
    subplot(3, 2, 5);
    imagesc(d1.amplitudes);
    title('20260612: Amplitude (counts)');
    colorbar;
    axis image;
    xlabel('X pixel'); ylabel('Y pixel');
    
    subplot(3, 2, 6);
    imagesc(d2.amplitudes);
    title('Paper: Amplitude (counts)');
    colorbar;
    axis image;
    xlabel('X pixel'); ylabel('Y pixel');
    
    % Adjust layout
    sgtitle('Comparison of SPAD 32x32 IRF Gaussian Fit Results', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Output file path
    out_file = 'C:\Users\27369\.gemini\antigravity\brain\b25eb207-5347-4d79-8fe8-4469104c5296\irf_comparison.png';
    
    % Save image
    print(fig, out_file, '-dpng', '-r150');
    close(fig);
    
    fprintf('Comparison plot successfully generated and saved to: %s\n', out_file);
end
