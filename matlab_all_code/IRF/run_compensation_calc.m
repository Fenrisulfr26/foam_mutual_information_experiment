function run_compensation_calc()
    % Define directories and files
    irf_dir = 'F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\IRF';
    input_file = fullfile(irf_dir, 'IRF_noLens_10avg_20260612_2210_fit_results.mat');
    output_file = fullfile(irf_dir, 'IRF_noLens_10avg_20260612_2210_compensation.mat');
    
    fprintf('Loading fit results from: %s\n', input_file);
    data = load(input_file);
    
    if ~isfield(data, 'peak_positions')
        error('Variable peak_positions not found in the input file!');
    end
    
    peak_positions = data.peak_positions;
    
    % Reference pixel coordinate is [16, 16] (1-based index)
    ref_row = 16;
    ref_col = 16;
    ref_val = peak_positions(ref_row, ref_col);
    
    fprintf('Reference pixel [%d, %d] peak position: %f bins\n', ref_row, ref_col, ref_val);
    
    % Step 1: Calculate relative deviation matrix (difference from reference pixel)
    relative_deviation = peak_positions - ref_val;
    
    % Step 2: Apply round() and negative (-) to generate final compensation matrix
    compensation_matrix = -round(relative_deviation);
    
    % Display some statistics
    fprintf('\n--- Statistics ---\n');
    fprintf('Relative Deviation:\n');
    fprintf('  Min: %f bins\n', min(relative_deviation(:)));
    fprintf('  Max: %f bins\n', max(relative_deviation(:)));
    fprintf('  Mean: %f bins\n', mean(relative_deviation(:)));
    fprintf('  Std Dev: %f bins\n', std(relative_deviation(:)));
    
    fprintf('\nCompensation Matrix:\n');
    fprintf('  Min compensation: %d bins\n', min(compensation_matrix(:)));
    fprintf('  Max compensation: %d bins\n', max(compensation_matrix(:)));
    fprintf('  Mean compensation: %.2f bins\n', mean(compensation_matrix(:)));
    
    % Save to MAT file
    save(output_file, 'peak_positions', 'relative_deviation', 'compensation_matrix', 'ref_row', 'ref_col', 'ref_val');
    fprintf('\nSaved compensation results to: %s\n', output_file);
end
