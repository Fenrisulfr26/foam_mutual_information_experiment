function my_display_hist(histgram, displayOptions)
%MY_DISPLAY_HIST Interactive viewer for a 3D histogram cube.
%
% Usage:
%   my_display_hist(histgram)
%   my_display_hist(histgram, displayOptions)
%
% Input:
%   histgram:
%       ny x nx x nBins matrix.
%
%       For your current case, it is usually:
%           32 x 32 x nBins
%
%       histgram(y, x, :) is the histogram curve of pixel (x, y).
%
% Display:
%   Left:
%       intensity image from sum(histgram, 3) or max(histgram, [], 3)
%
%   Right:
%       histogram curve of the pixel currently under the mouse.
%
% Note:
%   This function does NOT convert series to histogram.
%   Please run series2hist first.

    %% ---------------- Input check ----------------
    if nargin < 1
        error('Usage: my_display_hist(histgram)');
    end

    if nargin < 2 || isempty(displayOptions)
        displayOptions = struct();
    end

    if ndims(histgram) ~= 3
        error('histgram must be a 3D matrix, for example 32 x 32 x nBins.');
    end

    [ny, nx, nBins] = size(histgram);

    if ny < 1 || nx < 1 || nBins < 1
        error('histgram has invalid size.');
    end

    histgram = double(histgram);
    histgram(~isfinite(histgram)) = 0;

    if ~isfield(displayOptions, 'imageMode') || isempty(displayOptions.imageMode)
        displayOptions.imageMode = 'sum';
    end

    if ~isfield(displayOptions, 'smoothCurves') || isempty(displayOptions.smoothCurves)
        displayOptions.smoothCurves = false;
    end

    if ~isfield(displayOptions, 'smoothWindow') || isempty(displayOptions.smoothWindow)
        displayOptions.smoothWindow = 5;
    end

    imageMode = lower(string(displayOptions.imageMode));
    if imageMode == "total"
        imageMode = "sum";
    end

    if imageMode ~= "sum" && imageMode ~= "peak"
        error('displayOptions.imageMode must be "sum" or "peak".');
    end

    smoothCurves = logical(displayOptions.smoothCurves);
    smoothWindow = max(1, round(double(displayOptions.smoothWindow)));
    if mod(smoothWindow, 2) == 0
        smoothWindow = smoothWindow + 1;
    end

    if smoothCurves && smoothWindow > 1
        histgram = movmean(histgram, smoothWindow, 3);
    end

    %% ---------------- Gaussian fit display settings ----------------
    % Change this value to adjust the fitted Gaussian contour level.
    % 0.60 means offset + 60% of the fitted Gaussian amplitude.
    gaussianContourFraction = 0.60;

    %% ---------------- Prepare image and x-axis ----------------
    if imageMode == "peak"
        intensityImage = max(histgram, [], 3);
        intensityTitle = 'Peak photon count';
    else
        intensityImage = sum(histgram, 3);
        intensityTitle = 'Accumulated photon count';
    end

    if smoothCurves
        intensityTitle = sprintf('%s (smoothed)', intensityTitle);
    end

    [gaussianCenterX, gaussianCenterY, fittedGaussianImage, gaussianContourLevel] = ...
        fitGaussianCenter2D(intensityImage, gaussianContourFraction);

    % Since this function only receives histgram, no physical time axis is known.
    % Therefore, the x-axis is displayed as bin index.
    binAxis = 0:(nBins - 1);

    %% ---------------- Figure layout ----------------
    fig = figure( ...
        'Name', 'my_display_hist', ...
        'NumberTitle', 'off', ...
        'Color', 'w');

    set(fig, 'Position', [200, 200, 1050, 480]);

    axImg = axes( ...
        'Parent', fig, ...
        'Units', 'pixels', ...
        'Position', [70, 85, 360, 360], ...
        'Box', 'on');

    axCurve = axes( ...
        'Parent', fig, ...
        'Units', 'pixels', ...
        'Position', [550, 135, 430, 280], ...
        'Box', 'on');

    %% ---------------- Left image: accumulated intensity ----------------
    imagesc(axImg, intensityImage);
    axis(axImg, 'image');
    colormap(axImg, 'jet');
    colorbar(axImg);

    title(axImg, intensityTitle);
    xlabel(axImg, 'X pixel');
    ylabel(axImg, 'Y pixel');

    set(axImg, ...
        'XLim', [0.5, nx + 0.5], ...
        'YLim', [0.5, ny + 0.5]);

    hold(axImg, 'on');
    hMarker = plot(axImg, 1, 1, 'wo', ...
        'MarkerSize', 10, ...
        'LineWidth', 1.5);

    if isfinite(gaussianCenterX) && isfinite(gaussianCenterY)
        if ~isempty(fittedGaussianImage) && isfinite(gaussianContourLevel)
            contour(axImg, 1:nx, 1:ny, fittedGaussianImage, ...
                [gaussianContourLevel, gaussianContourLevel], ...
                'k-', ...
                'LineWidth', 1);
        end

        plot(axImg, gaussianCenterX, gaussianCenterY, 'wx', ...
            'MarkerSize', 8, ...
            'LineWidth', 1);
    end
    hold(axImg, 'off');

    %% ---------------- Right curve: selected pixel histogram ----------------
    hCurve = plot(axCurve, binAxis, zeros(1, nBins), 'LineWidth', 1.5);

    grid(axCurve, 'on');

    if nBins > 1
        xlim(axCurve, [binAxis(1), binAxis(end)]);
    else
        xlim(axCurve, [0, 1]);
    end
    
    xlabel(axCurve, 'Bin index');
    ylabel(axCurve, 'Counts');
    title(axCurve, 'Pixel histogram');

    %% ---------------- Text information ----------------
    txtX = uicontrol(fig, ...
        'Style', 'text', ...
        'String', 'X: 1', ...
        'BackgroundColor', 'w', ...
        'FontSize', 10, ...
        'HorizontalAlignment', 'left', ...
        'Position', [550, 85, 90, 25]);

    txtY = uicontrol(fig, ...
        'Style', 'text', ...
        'String', 'Y: 1', ...
        'BackgroundColor', 'w', ...
        'FontSize', 10, ...
        'HorizontalAlignment', 'left', ...
        'Position', [650, 85, 90, 25]);

    txtSum = uicontrol(fig, ...
        'Style', 'text', ...
        'String', 'Total counts: 0', ...
        'BackgroundColor', 'w', ...
        'FontSize', 10, ...
        'HorizontalAlignment', 'left', ...
        'Position', [750, 85, 200, 25]);

    %% ---------------- Mouse callback ----------------
    lastX = NaN;
    lastY = NaN;

    set(fig, 'WindowButtonMotionFcn', @mouseMoveCallback);

    % Show pixel (1,1) by default
    updatePixel(1, 1);

    %% ============================================================
    % Nested callback functions
    % ============================================================

    function mouseMoveCallback(~, ~)
        pt = get(axImg, 'CurrentPoint');

        xpos = round(pt(1, 1));
        ypos = round(pt(1, 2));

        % Only update when mouse is inside the image region
        if xpos < 1 || xpos > nx || ypos < 1 || ypos > ny
            return;
        end

        % Avoid repeated redraw for the same pixel
        if xpos == lastX && ypos == lastY
            return;
        end

        updatePixel(xpos, ypos);
    end

    function updatePixel(xpos, ypos)
        lastX = xpos;
        lastY = ypos;

        selectedCurve = squeeze(histgram(ypos, xpos, :));
        selectedCurve = selectedCurve(:).';

        totalCounts = sum(selectedCurve);
        peakCounts = max(selectedCurve);

        set(hCurve, 'XData', binAxis, 'YData', selectedCurve);

        % ymax = max(selectedCurve);
        ymax = max(histgram,[],"all");

        if isempty(ymax) || ymax <= 0
            ylim(axCurve, [0, 1]);
        else
            ylim(axCurve, [0, ymax * 1.1]);
        end

        title(axCurve, sprintf('Histogram at X = %d, Y = %d', xpos, ypos));

        set(txtX, 'String', sprintf('X: %d', xpos));
        set(txtY, 'String', sprintf('Y: %d', ypos));
        set(txtSum, 'String', sprintf('Total: %.3g   Peak: %.3g', totalCounts, peakCounts));

        set(hMarker, 'XData', xpos, 'YData', ypos);

        drawnow limitrate;
    end
end

function [centerX, centerY, fittedImage, contourLevel] = fitGaussianCenter2D(intensityImage, contourFraction)
%FITGAUSSIANCENTER2D Estimate the center of a 2D Gaussian on an image.
% Uses fminsearch so the viewer does not depend on toolboxes.

    centerX = NaN;
    centerY = NaN;
    fittedImage = [];
    contourLevel = NaN;

    if nargin < 2 || ~isfinite(contourFraction)
        contourFraction = 0.60;
    end
    contourFraction = min(max(contourFraction, 0), 1);

    imageData = double(intensityImage);
    imageData(~isfinite(imageData)) = 0;

    [ny, nx] = size(imageData);
    if ny < 1 || nx < 1
        return;
    end

    minValue = min(imageData(:));
    signalData = imageData - minValue;
    signalSum = sum(signalData(:));
    if signalSum <= 0
        [~, maxIdx] = max(imageData(:));
        [centerY, centerX] = ind2sub([ny, nx], maxIdx);
        return;
    end

    [xGrid, yGrid] = meshgrid(1:nx, 1:ny);
    x0 = sum(xGrid(:) .* signalData(:)) / signalSum;
    y0 = sum(yGrid(:) .* signalData(:)) / signalSum;

    xVariance = sum(((xGrid(:) - x0) .^ 2) .* signalData(:)) / signalSum;
    yVariance = sum(((yGrid(:) - y0) .^ 2) .* signalData(:)) / signalSum;

    sigmaX0 = max(sqrt(max(xVariance, eps)), 1);
    sigmaY0 = max(sqrt(max(yVariance, eps)), 1);
    offset0 = minValue;
    amplitude0 = max(imageData(:)) - offset0;
    if amplitude0 <= 0
        amplitude0 = max(signalData(:));
    end
    amplitude0 = max(amplitude0, eps);

    initialParams = [offset0, log(amplitude0), x0, y0, log(sigmaX0), log(sigmaY0)];

    objective = @(params) sum((imageData(:) - gaussianModel(params, xGrid, yGrid)).^2);
    options = optimset('Display', 'off', 'MaxIter', 1000, 'MaxFunEvals', 3000);

    try
        fittedParams = fminsearch(objective, initialParams, options);
        centerX = min(max(fittedParams(3), 1), nx);
        centerY = min(max(fittedParams(4), 1), ny);
    catch
        fittedParams = initialParams;
        centerX = min(max(x0, 1), nx);
        centerY = min(max(y0, 1), ny);
    end

    fittedImage = reshape(gaussianModel(fittedParams, xGrid, yGrid), ny, nx);
    contourLevel = fittedParams(1) + contourFraction * exp(fittedParams(2));
end

function values = gaussianModel(params, xGrid, yGrid)
    offset = params(1);
    amplitude = exp(params(2));
    centerX = params(3);
    centerY = params(4);
    sigmaX = max(exp(params(5)), eps);
    sigmaY = max(exp(params(6)), eps);

    values = offset + amplitude .* exp(-0.5 .* ( ...
        ((xGrid(:) - centerX) ./ sigmaX) .^ 2 + ...
        ((yGrid(:) - centerY) ./ sigmaY) .^ 2));
end
