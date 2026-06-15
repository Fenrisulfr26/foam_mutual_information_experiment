%% Pixel size measurement: center-pixel total intensity map
% Reads a 25 x 25 pixel-size scan folder, sums hist(16,16,:) for every
% scan point, and saves a 25 x 25 total-intensity map.

clear; clc; close all;

projectRoot = fileparts(mfilename('fullpath'));
scanDir = fullfile(projectRoot, 'data', 'Pixel size measurement', ...
    'pixel_size_measurement_20260613_174324_25x25');

pixelRow = 16;
pixelCol = 16;
expectedGridSize = [25, 25];

pointFiles = dir(fullfile(scanDir, 'hist_point*_row*_col*.mat'));
if isempty(pointFiles)
    error('No scan point files found in: %s', scanDir);
end

totalIntensity = nan(expectedGridSize);
pointFileMap = strings(expectedGridSize);

for iFile = 1:numel(pointFiles)
    fileName = pointFiles(iFile).name;
    token = regexp(fileName, 'hist_point(\d+)_row(\d+)_col(\d+)_', 'tokens', 'once');

    if isempty(token)
        warning('Skipping unexpected file name: %s', fileName);
        continue;
    end

    scanRow = str2double(token{2});
    scanCol = str2double(token{3});

    if scanRow < 1 || scanRow > expectedGridSize(1) || ...
            scanCol < 1 || scanCol > expectedGridSize(2)
        warning('Skipping out-of-grid point: %s', fileName);
        continue;
    end

    data = load(fullfile(scanDir, fileName), 'hist');
    if ~isfield(data, 'hist')
        error('Missing variable "hist" in: %s', fileName);
    end

    histData = double(data.hist);
    if ndims(histData) ~= 3
        error('Expected 3-D hist in %s, got size %s.', fileName, mat2str(size(histData)));
    end
    if pixelRow > size(histData, 1) || pixelCol > size(histData, 2)
        error('Requested pixel (%d,%d) is outside hist size %s in %s.', ...
            pixelRow, pixelCol, mat2str(size(histData)), fileName);
    end

    totalIntensity(scanRow, scanCol) = sum(histData(pixelRow, pixelCol, :), 'all');
    pointFileMap(scanRow, scanCol) = string(fileName);
end

missingMask = isnan(totalIntensity);
if any(missingMask, 'all')
    [missingRows, missingCols] = find(missingMask);
    warning('Missing %d grid points. First missing point is row %d, col %d.', ...
        numel(missingRows), missingRows(1), missingCols(1));
end

result = struct();
result.scanDir = scanDir;
result.pixelRow = pixelRow;
result.pixelCol = pixelCol;
result.gridSize = expectedGridSize;
result.totalIntensity = totalIntensity;
result.pointFileMap = pointFileMap;

matPath = fullfile(scanDir, sprintf('total_intensity_pixel_r%02d_c%02d_25x25.mat', pixelRow, pixelCol));
csvPath = fullfile(scanDir, sprintf('total_intensity_pixel_r%02d_c%02d_25x25.csv', pixelRow, pixelCol));
pngPath = fullfile(scanDir, sprintf('total_intensity_pixel_r%02d_c%02d_25x25.png', pixelRow, pixelCol));

save(matPath, 'result');
writematrix(totalIntensity, csvPath);

fig = figure('Color', 'w', 'Name', 'Pixel size measurement total intensity');
imagesc(totalIntensity);
axis image;
set(gca, 'YDir', 'normal');
colormap(parula);
colorbar;
xlabel('Scan column');
ylabel('Scan row');
title(sprintf('Total intensity of hist(%d,%d,:) over 25 x 25 scan', pixelRow, pixelCol), ...
    'Interpreter', 'none');
exportgraphics(fig, pngPath, 'Resolution', 300);

fprintf('Processed %d files from:\n  %s\n', numel(pointFiles), scanDir);
fprintf('Summed pixel: hist(%d,%d,:) -> total intensity\n', pixelRow, pixelCol);
fprintf('Saved MAT: %s\n', matPath);
fprintf('Saved CSV: %s\n', csvPath);
fprintf('Saved PNG: %s\n', pngPath);
