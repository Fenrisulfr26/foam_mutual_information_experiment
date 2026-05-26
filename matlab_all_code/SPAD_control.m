%% SPAD setup
addpath(genpath('D:\coding\Photon Force\PF32\Matlab\my_wrapper'))
% Make sure you run this function just once and keep the "pf32" pointer
% Close the device by calling "pf_open" or restart Matlab before calling
% again.
OK = pf_getLibraryStatus;
if(OK)
    % Close the library if it is already open. The scripts will load it
    % when needed.
    pf_close
end
[pf32, alias, report] = pf_open;
pf_getLibraryStatus;

%%

close('all')
clc
addpath(genpath('D:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code'))

% ---------------- Acquisition settings ----------------
pf_setMode(pf32, 'TCSPC_laser_master')

pf_setNumberAccumulations(pf32, 1)

expo_time = 2;   % us or camera-defined unit
pf_setExposure(pf32, expo_time)

laser_rep_rate = 80.33e6;   % Hz

num_of_frames = 1e5;        % frames per group
num_groups = 50;             % number of repeated acquisitions, change freely
issaverepeat = 0;

fprintf('Start acquisition: %d groups, %d frames per group.\n', ...
    num_groups, num_of_frames)

hist_sum = [];
hist_all = cell(num_groups, 1);

tic

for group_idx = 1:1

    fprintf('\nAcquiring group %d / %d ...\n', group_idx, num_groups)

    series = pf_getMultipleFrames(pf32, num_of_frames);

    % Optionally rotate image to account for lens inversion
    % series = rot90(series, 2);

    hist_current = series2hist(series, laser_rep_rate);

    % Convert to double before accumulation to avoid integer overflow
    % hist_current = double(hist_current);
    % 
    % if issaverepeat == 1
    %     hist = hist_current;
    %     save(sprintf("data/test_repeat/hist_%d",group_idx),"hist")
    % end
    % 
    % if group_idx == 1
    %     hist_sum = zeros(size(hist_current));
    % else
    %     if ~isequal(size(hist_current), size(hist_sum))
    %         error('Histogram size mismatch at group %d.', group_idx)
    %     end
    % end

    % hist_sum = hist_sum + hist_current;
    % hist_all{group_idx} = hist_current;

    % clear series hist_current

end

elapsed_time = toc;

% hist_avg = hist_sum / num_groups;

fprintf('\nFinished acquisition.\n')
fprintf('Total time: %.2f s\n', elapsed_time)
fprintf('Average time per group: %.2f s\n', elapsed_time / num_groups)

% ---------------- Display averaged histogram ----------------
hist_current = correct_hot_dark_pixels(hist_current);
my_display_hist(hist_current)


% save('IRF.mat',"hist_current")
% my_display_hist(hist_avg)
% my_display_hist(hist_sum)

%%

pf_close(pf32);

%% check multiple hists
figure
hist_sum = zeros(32,32,227) ;

for i = 1:9
    load(sprintf("D:\\OneDrive\\foam_imaging_project\\experiment_setup\\matlab_all_code\\data\\test_repeat\\hist_%d.mat",i))
    subplot(3,3,i)
    plot(squeeze(hist(16,16,:)))
    ylim([0,300])
    hist_sum = hist_sum + hist;
end

my_display_hist(rebin_hist_time(hist_sum/9,2))

