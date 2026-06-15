function plot_skew_correlation()
    % Get the current script folder as base
    PROJECT_ROOT = fileparts(mfilename('fullpath'));
    irf_dir = PROJECT_ROOT;
    
    file_nolens = fullfile(irf_dir, 'IRF_noLens_10avg_20260612_2210_fit_results.mat');
    file_paper = fullfile(irf_dir, 'IRF_nolens_paper_10_avg_20261712_220615_fit_results.mat');
    
    if ~exist(file_nolens, 'file') || ~exist(file_paper, 'file')
        error('Required fit results files do not exist!');
    end
    
    d_nolens = load(file_nolens);
    d_paper = load(file_paper);
    
    P_nolens = d_nolens.peak_positions;
    P_paper = d_paper.peak_positions;
    
    % Filter out NaNs if any (both should have 1024 valid pixels)
    valid_idx = ~isnan(P_nolens) & ~isnan(P_paper);
    
    v_nolens = P_nolens(valid_idx);
    v_paper = P_paper(valid_idx);
    
    % 1. Calculate correlation coefficient
    R = corrcoef(v_paper, v_nolens);
    r_val = R(1, 2);
    
    % 2. Fit a line: P_nolens = slope * P_paper + intercept
    p_fit = polyfit(v_paper, v_nolens, 1);
    slope = p_fit(1);
    intercept = p_fit(2);
    
    % 3. Calculate difference
    diff_map = P_nolens - P_paper;
    v_diff = diff_map(valid_idx);
    mean_diff = mean(v_diff);
    std_diff = std(v_diff);
    
    % Set up figure (invisible)
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 1000]);
    
    % Subplot 1: 2D Map of peak positions for noLens
    subplot(2, 2, 1);
    imagesc(P_nolens);
    title('noLens 10avg: Peak Positions (\mu, bins)');
    colorbar;
    axis image;
    xlabel('X pixel'); ylabel('Y pixel');
    
    % Subplot 2: 2D Map of peak positions for Paper
    subplot(2, 2, 2);
    imagesc(P_paper);
    title('Paper 10avg: Peak Positions (\mu, bins)');
    colorbar;
    axis image;
    xlabel('X pixel'); ylabel('Y pixel');
    
    % Subplot 3: Scatter Plot & Correlation
    subplot(2, 2, 3);
    scatter(v_paper, v_nolens, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    x_range = linspace(min(v_paper), max(v_paper), 100);
    y_fit = polyval(p_fit, x_range);
    plot(x_range, y_fit, 'r-', 'LineWidth', 2);
    
    title('Peak Position Correlation (Pixel-by-Pixel)');
    xlabel('Paper Peak Position (bins)');
    ylabel('noLens Peak Position (bins)');
    grid on;
    
    % Add text info
    info_str = sprintf('Pearson r = %.5f\nSlope = %.4f\nIntercept = %.4f bins\nMean Diff = %.4f bins\nStd Diff = %.4f bins', ...
        r_val, slope, intercept, mean_diff, std_diff);
    text(min(v_paper) + 0.5, max(v_nolens) - 2.5, info_str, 'FontSize', 11, 'BackgroundColor', 'w', 'EdgeColor', 'k');
    
    % Subplot 4: 2D Map of Difference (noLens - Paper)
    subplot(2, 2, 4);
    imagesc(diff_map);
    title('Difference Map: noLens - Paper (bins)');
    colorbar;
    axis image;
    xlabel('X pixel'); ylabel('Y pixel');
    
    % Main title
    sgtitle('Timing Skew Correlation Analysis (With vs. Without Paper)', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Output file path
    out_file = 'C:\Users\27369\.gemini\antigravity\brain\b25eb207-5347-4d79-8fe8-4469104c5296\skew_correlation.png';
    
    % Save image
    print(fig, out_file, '-dpng', '-r150');
    close(fig);
    
    % Output text report
    report_file = fullfile(irf_dir, 'correlation_report.txt');
    fid = fopen(report_file, 'w');
    fprintf(fid, '==================================================\n');
    fprintf(fid, 'Peak Position Correlation Report (With vs. Without Paper)\n');
    fprintf(fid, '==================================================\n');
    fprintf(fid, 'Pearson Correlation Coefficient (r)       = %.6f\n', r_val);
    fprintf(fid, 'Coefficient of Determination (R^2)        = %.6f\n', r_val^2);
    fprintf(fid, 'Linear Fit Equation                       = noLens = %.4f * Paper + %.4f\n', slope, intercept);
    fprintf(fid, 'Mean difference (noLens - Paper)          = %.4f bins\n', mean_diff);
    fprintf(fid, 'Standard deviation of difference (residual) = %.4f bins\n', std_diff);
    fprintf(fid, '--------------------------------------------------\n');
    fclose(fid);
    
    fprintf('Correlation analysis completed. Plot saved to: %s\n', out_file);
end
