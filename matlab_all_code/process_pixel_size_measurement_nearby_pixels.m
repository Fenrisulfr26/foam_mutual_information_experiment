%% Pixel size measurement: nearby-pixel total intensity maps
% Reads a 25 x 25 pixel-size scan folder and saves total-intensity maps for
% selected SPAD pixels. Each map is sum(hist(pixelRow,pixelCol,:)).

clear; clc; close all;

projectRoot = fileparts(mfilename('fullpath'));
scanDir = fullfile(projectRoot, 'data', 'Pixel size measurement', ...
    'pixel_size_measurement_20260613_174324_25x25');

selectedPixels = [
    16, 16
    16, 17
    17, 16
    17, 17
    ];

expectedGridSize = [25, 25];

pointFiles = dir(fullfile(scanDir, 'hist_point*_row*_col*.mat'));
if isempty(pointFiles)
    error('No scan point files found in: %s', scanDir);
end

numPixels = size(selectedPixels, 1);
allMaps = nan([expectedGridSize, numPixels]);
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

    for iPixel = 1:numPixels
        pixelRow = selectedPixels(iPixel, 1);
        pixelCol = selectedPixels(iPixel, 2);

        if pixelRow > size(histData, 1) || pixelCol > size(histData, 2)
            error('Requested pixel (%d,%d) is outside hist size %s in %s.', ...
                pixelRow, pixelCol, mat2str(size(histData)), fileName);
        end

        allMaps(scanRow, scanCol, iPixel) = sum(histData(pixelRow, pixelCol, :), 'all');
    end

    pointFileMap(scanRow, scanCol) = string(fileName);
end

missingMask = isnan(allMaps(:, :, 1));
if any(missingMask, 'all')
    [missingRows, missingCols] = find(missingMask);
    warning('Missing %d grid points. First missing point is row %d, col %d.', ...
        numel(missingRows), missingRows(1), missingCols(1));
end

globalMin = min(allMaps, [], 'all', 'omitnan');
globalMax = max(allMaps, [], 'all', 'omitnan');
if isempty(globalMin) || isempty(globalMax) || globalMin == globalMax
    globalMin = 0;
    globalMax = 1;
end

summary = struct();
summary.scanDir = scanDir;
summary.selectedPixels = selectedPixels;
summary.gridSize = expectedGridSize;
summary.allMaps = allMaps;
summary.pointFileMap = pointFileMap;

summaryMatPath = fullfile(scanDir, 'total_intensity_nearby_pixels_summary.mat');
save(summaryMatPath, 'summary');

figMontage = figure('Color', 'w', 'Name', 'Nearby pixel total intensity maps');
tiledlayout(figMontage, 1, numPixels, 'TileSpacing', 'compact', 'Padding', 'compact');

for iPixel = 1:numPixels
    pixelRow = selectedPixels(iPixel, 1);
    pixelCol = selectedPixels(iPixel, 2);
    totalIntensity = allMaps(:, :, iPixel);

    result = struct();
    result.scanDir = scanDir;
    result.pixelRow = pixelRow;
    result.pixelCol = pixelCol;
    result.gridSize = expectedGridSize;
    result.totalIntensity = totalIntensity;
    result.pointFileMap = pointFileMap;

    stem = sprintf('total_intensity_pixel_r%02d_c%02d_25x25', pixelRow, pixelCol);
    matPath = fullfile(scanDir, [stem, '.mat']);
    csvPath = fullfile(scanDir, [stem, '.csv']);
    pngPath = fullfile(scanDir, [stem, '.png']);

    save(matPath, 'result');
    writematrix(totalIntensity, csvPath);

    fig = figure('Color', 'w', 'Name', stem);
    imagesc(totalIntensity);
    axis image;
    set(gca, 'YDir', 'normal');
    colormap(parula);
    colorbar;
    clim([globalMin, globalMax]);
    xlabel('Scan column');
    ylabel('Scan row');
    title(sprintf('Total intensity of hist(%d,%d,:)', pixelRow, pixelCol), ...
        'Interpreter', 'none');
    exportgraphics(fig, pngPath, 'Resolution', 300);
    close(fig);

    nexttile;
    imagesc(totalIntensity);
    axis image;
    set(gca, 'YDir', 'normal');
    clim([globalMin, globalMax]);
    title(sprintf('hist(%d,%d,:)', pixelRow, pixelCol), 'Interpreter', 'none');
    xlabel('Scan column');
    ylabel('Scan row');
end

colormap(figMontage, parula);
colorbar;
montagePath = fullfile(scanDir, 'total_intensity_nearby_pixels_montage.png');
exportgraphics(figMontage, montagePath, 'Resolution', 300);

fprintf('Processed %d files from:\n  %s\n', numel(pointFiles), scanDir);
fprintf('Saved summary MAT: %s\n', summaryMatPath);
fprintf('Saved montage PNG: %s\n', montagePath);
for iPixel = 1:numPixels
    pixelRow = selectedPixels(iPixel, 1);
    pixelCol = selectedPixels(iPixel, 2);
    pixelMap = allMaps(:, :, iPixel);
    fprintf('hist(%d,%d,:): min=%.0f max=%.0f mean=%.2f\n', ...
        pixelRow, pixelCol, min(pixelMap, [], 'all'), ...
        max(pixelMap, [], 'all'), mean(pixelMap, 'all', 'omitnan'));
end
