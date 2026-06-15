%% SPAD Setup & Initialization
% This section dynamically sets up paths and initializes the Photon Force PF32 camera.
% It should be run once at the beginning of your MATLAB session.

clc;
close all;

% Get the project directory dynamically to ensure portability
PROJECT_ROOT = fileparts(mfilename('fullpath'));
addpath(genpath(PROJECT_ROOT));

% Search standard library paths for the Photon Force PF32 Matlab wrapper
PF_WRAPPER_PATHS = {
    'D:\coding\Photon Force\PF32\Matlab\my_wrapper'
    'D:\coding\PF32\Matlab'
    };

pf_path_found = false;
for pathIdx = 1:numel(PF_WRAPPER_PATHS)
    if isfolder(PF_WRAPPER_PATHS{pathIdx})
        addpath(genpath(PF_WRAPPER_PATHS{pathIdx}));
        pf_path_found = true;
        break;
    end
end

if ~pf_path_found
    warning('Photon Force PF32 wrapper library path not found. Please ensure PF32 Matlab SDK wrapper is installed.');
end

% Safely check and reset the library connection before opening the camera
try
    if pf_getLibraryStatus()
        pf_close();
    end
catch ME
    warning('SPAD library status check failed: %s', ME.message);
end

% Open connection and fetch device pointer
[pf32, alias, report] = pf_open();
pf_getLibraryStatus();

fprintf('SPAD camera successfully initialized.\n');


%% Acquisition Configuration & Execution
% Set acquisition parameters, run the acquisition loop, display and save results.

% --- Reset State (Optional) ---
close all;
clc;

% --- Acquisition Parameters ---
mode = 'TCSPC_laser_master';
num_accumulations = 1;
expo_time_us = 2;             % Exposure time (us or camera-defined unit)
laser_rep_rate = 80.33e6;        % Laser repetition frequency (Hz)
num_of_frames = 1e5;          % Frames per acquisition group
num_groups = 10;              % Number of groups to acquire and average (change freely)
save_individual_repeats = false; % Set to true to save individual group histograms to disk

% Apply camera configuration
pf_setMode(pf32, mode);
pf_setNumberAccumulations(pf32, num_accumulations);
pf_setExposure(pf32, expo_time_us);

fprintf('Starting acquisition: %d groups, %d frames per group.\n', num_groups, num_of_frames);

% Preallocate variables for tracking and accumulation
hist_all = cell(num_groups, 1);
hist_sum = [];

tic;

for group_idx = 1:num_groups
    fprintf('\nAcquiring group %d / %d ...\n', group_idx, num_groups);

    % Capture frames from SPAD
    series = pf_getMultipleFrames(pf32, num_of_frames);

    % Optional: Rotate image to account for physical lens inversion
    % series = rot90(series, 2);

    % Convert raw frame series to histogram data
    hist_current = series2hist(series, laser_rep_rate);
    hist_current = double(hist_current); % Convert to double to avoid integer overflow

    % Initialize sum on first iteration
    if isempty(hist_sum)
        hist_sum = zeros(size(hist_current));
    else
        if ~isequal(size(hist_current), size(hist_sum))
            error('Histogram size mismatch at group %d. Expected %s, got %s.', ...
                group_idx, mat2str(size(hist_sum)), mat2str(size(hist_current)));
        end
    end

    % Accumulate histogram
    hist_sum = hist_sum + hist_current;
    hist_all{group_idx} = hist_current;

    % Save individual repeated acquisitions if requested
    if save_individual_repeats
        repeat_dir = fullfile(pwd, 'data', 'test_repeat');
        if ~exist(repeat_dir, 'dir')
            mkdir(repeat_dir);
        end
        save_path = fullfile(repeat_dir, sprintf('hist_%d.mat', group_idx));
        % Save as 'hist' variable for backward compatibility with check script
        hist = hist_current; %#ok<NASGU>
        save(save_path, 'hist');
    end
end

elapsed_time = toc;

% Calculate average histogram across all groups
hist_avg = hist_sum / num_groups;

fprintf('\nFinished acquisition.\n');
fprintf('  Total time: %.2f s\n', elapsed_time);
fprintf('  Average time per group: %.2f s\n', elapsed_time / num_groups);

% --- Data Processing & Display ---
% Correct dark/hot pixels on the averaged histogram
[hist_corrected, acq_info] = correct_hot_dark_pixels(hist_avg);

% Display the corrected histogram
my_display_hist(hist_corrected);

% Display a maximum-intensity projection map of the final acquired series
figure('Name', 'Final Acquisition Max Projection');
imagesc(max(series, [], 3));
colorbar;
title('Max-Intensity Projection of Final Series');

% --- Save Option ---
% Prompt the user to choose whether to save the acquired histogram data
save_choice = questdlg('是否保存采集后的直方图数据？', '保存采集结果', '保存', '不保存', '保存');

if strcmp(save_choice, '保存')
    % Suggest a default timestamped filename in the current folder (pwd)
    default_name = fullfile(pwd, sprintf('acq_hist_%s.mat', datestr(now, 'yyyyMMdd_HHmmss')));
    [file_name, file_path] = uiputfile('*.mat', '选择保存位置', default_name);

    if ischar(file_name)
        full_save_path = fullfile(file_path, file_name);

        % Save variables: save corrected hist as 'hist' for compatibility with display scripts
        hist = hist_corrected; %#ok<NASGU>
        hist_raw = hist_avg; %#ok<NASGU>
        info = acq_info; %#ok<NASGU>
        % Add acquisition metadata to info struct
        info.num_groups = num_groups;
        info.num_of_frames = num_of_frames;
        info.expo_time_us = expo_time_us;
        info.laser_rep_rate = laser_rep_rate;
        info.hist_all = hist_all;

        save(full_save_path, 'hist', 'hist_raw', 'info', 'series');
        series_filename = full_save_path;
        fprintf('采集数据已成功保存至：%s\n', full_save_path);
    else
        fprintf('用户取消了保存操作。\n');
    end
else
    fprintf('选择不保存本次采集数据。\n');
end


%% Single Group Acquisition & Dark Noise Calibration
% This section acquires a single group of frames (useful for dark noise reference or single tests),
% performs hot/dark pixel calibration, visualizes the effect, and prompts the user to save it.

% Ensure PROJECT_ROOT exists in the workspace
if ~exist('PROJECT_ROOT', 'var')
    PROJECT_ROOT = fileparts(mfilename('fullpath'));
end

% Ensure SPAD is connected
if ~exist('pf32', 'var') || isempty(pf32)
    error('SPAD camera pointer "pf32" does not exist. Please initialize the hardware first.');
end

% --- Configuration ---
calib_expo_time_us = 2;         % Exposure time (us)
calib_laser_rep_rate = 80e6;     % Laser repetition frequency (Hz)
calib_num_frames = 1e5;         % Number of frames to acquire

% Apply camera configuration
pf_setMode(pf32, 'TCSPC_laser_master');
pf_setNumberAccumulations(pf32, 1);
pf_setExposure(pf32, calib_expo_time_us);

fprintf('Acquiring 1 group (%d frames) for calibration...\n', calib_num_frames);

% Capture single group
calib_series = pf_getMultipleFrames(pf32, calib_num_frames);

% Convert to histogram (raw)
calib_hist_raw = series2hist(calib_series, calib_laser_rep_rate);
calib_hist_raw = double(calib_hist_raw);

% Perform hot/dark pixel correction
[calib_hist_corrected, calib_info] = correct_hot_dark_pixels(calib_hist_raw);

% --- Visualization ---
% Observe the corrected histogram
my_display_hist(calib_hist_corrected);

% --- Save Option ---
% Prompt the user to choose whether to save the calibrated histogram data
save_choice = questdlg('是否保存校准后的直方图数据？', '保存校准结果', '保存', '不保存', '保存');

if strcmp(save_choice, '保存')
    % Suggest a default timestamped filename in the current folder (pwd)
    default_name = fullfile(pwd, sprintf('calib_hist_%s.mat', datestr(now, 'yyyyMMdd_HHmmss')));
    [file_name, file_path] = uiputfile('*.mat', '选择保存位置', default_name);

    if ischar(file_name)
        full_save_path = fullfile(file_path, file_name);

        % Save variables: save corrected hist as 'hist' for compatibility with display scripts
        hist = calib_hist_corrected; %#ok<NASGU>
        hist_raw = calib_hist_raw; %#ok<NASGU>
        info = calib_info; %#ok<NASGU>
        series = calib_series; %#ok<NASGU>

        save(full_save_path, 'hist', 'hist_raw', 'info', 'series');
        fprintf('校准数据已成功保存至：%s\n', full_save_path);
    else
        fprintf('用户取消了保存操作。\n');
    end
else
    fprintf('选择不保存本次校准数据。\n');
end


%% Close SPAD Connection
% Run this section to disconnect from the camera safely.

if exist('pf32', 'var')
    pf_close(pf32);
    fprintf('SPAD camera connection closed successfully.\n');
else
    fprintf('No active camera connection pointer (pf32) found.\n');
end


%% Verification: Analyze Repeated Histograms
% Read and plot the histogram at pixel (16,16) from the saved repeat runs.

% Ensure PROJECT_ROOT exists in the workspace
if ~exist('PROJECT_ROOT', 'var')
    PROJECT_ROOT = fileparts(mfilename('fullpath'));
end

repeat_dir = fullfile(pwd, 'data', 'test_repeat');
check_hist_sum = [];
num_files_to_check = 9;
loaded_count = 0;

figure('Name', 'Repeat Histograms (Pixel 16,16)');

for i = 1:num_files_to_check
    file_path = fullfile(repeat_dir, sprintf('hist_%d.mat', i));

    if exist(file_path, 'file')
        data = load(file_path, 'hist');
        if isfield(data, 'hist')
            h = double(data.hist);

            loaded_count = loaded_count + 1;
            if isempty(check_hist_sum)
                check_hist_sum = zeros(size(h));
            end
            check_hist_sum = check_hist_sum + h;

            subplot(3, 3, loaded_count);
            plot(squeeze(h(16, 16, :)));
            ylim([0, 300]);
            title(sprintf('Group %d', i));
            grid on;
        end
    else
        warning('Repeat file not found: %s', file_path);
    end
end

if loaded_count > 0
    hist_check_avg = check_hist_sum / loaded_count;

    % Display averaged and time-rebinned histogram
    try
        hist_rebinned = rebin_hist_time(hist_check_avg, 2);
        my_display_hist(hist_rebinned);
    catch ME
        warning('Failed to time-rebin or display the histogram: %s', ME.message);
        my_display_hist(hist_check_avg);
    end
else
    fprintf('No repeat files found in %s to analyze.\n', repeat_dir);
end


%% Verification: Check Series Distribution
% Load photon arrival series data and plot the distribution of arrival bins.

% Ensure PROJECT_ROOT exists in the workspace
if ~exist('PROJECT_ROOT', 'var')
    PROJECT_ROOT = fileparts(mfilename('fullpath'));
end

% Try loading from:
% 1. The run we just performed (series_filename)
% 2. A default local series.mat file
% 3. The latest series_*.mat file in the series directory
loaded_series = false;

if exist('series_filename', 'var') && exist(series_filename, 'file')
    load(series_filename, 'series');
    fprintf('Loaded series from current session: %s\n', series_filename);
    loaded_series = true;
elseif exist('series.mat', 'file')
    load('series.mat', 'series');
    fprintf('Loaded default "series.mat" from current folder.\n');
    loaded_series = true;
else
    series_dir = fullfile(pwd, 'series');
    if exist(series_dir, 'dir')
        files = dir(fullfile(series_dir, 'series_*.mat'));
        if ~isempty(files)
            [~, latest_idx] = max([files.datenum]);
            latest_file = fullfile(series_dir, files(latest_idx).name);
            fprintf('Loaded latest timestamped series file: %s\n', latest_file);
            load(latest_file, 'series');
            loaded_series = true;
        end
    end
end

if loaded_series && exist('series', 'var')
    % Analyze center pixel (1, 1) timing codes
    center_pixel_data = squeeze(series(1, 1, :));
    center_pixel_data(center_pixel_data == 0) = []; % Remove padding/invalid codes

    if ~isempty(center_pixel_data)
        figure('Name', 'Photon Arrival TDC Bin Distribution');
        histogram(center_pixel_data, 236);
        title('TDC Bin Code Distribution at Pixel (1,1)');
        xlabel('TDC Bin Code');
        ylabel('Photon Event Counts');
        grid on;

        fprintf('Maximum photon arrival TDC code recorded: %d\n', max(center_pixel_data));
    else
        fprintf('No valid (non-zero) photon events found at pixel (1,1).\n');
    end
else
    fprintf('No series data file was found to check.\n');
end
