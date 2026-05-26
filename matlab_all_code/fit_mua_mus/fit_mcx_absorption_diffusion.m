%% Fit absorption coefficient and diffusion coefficient using MCX
% Target curve: hist_5us_1e+06_38.5deg_obj.mat, pixel (16,16)
% IRF kernel:   IRF.mat, pixel (16,16)
%
% Parameter convention:
%   mua: absorption coefficient, in 1/mm
%   D:   diffusion coefficient, in mm
%
% MCX uses mua and mus. This script converts D to reduced scattering
% coefficient musp using D = 1/(3*(mua + musp)), then mus = musp/(1 - g).

clear; clc;

%% ---------------- User configuration ----------------
mcxlabPath = 'F:\OneDrive - University of Glasgow\project in UK 2024\MCX\MCXStudio\MATLAB\mcxlab';
targetFile = fullfile(pwd, 'data', '50mm_60d_foam', 'hist_5us_1e+06_38.5deg_obj.mat');
irfFile = fullfile(pwd, 'data', 'IRF.mat');

pixelYX = [16, 16];          % hist(y,x,t), requested pixel

sampleSize_mm = [250, 250, 50];  % [x,y,z] in mm
voxelSize_mm = 2.5;              % 250x250x50 mm -> 100x100x20 voxels
nMedium = 1.48;
g = 1;

sourceAngle_deg = 38.5;
sourcePosition_mm = [125, 125, 2.5];       % top voxel center
detectorPosition_mm = [125, 125, 50];      % bottom center, transmission geometry
detectorRadius_mm = 8;

% Histogram axis. Existing series2hist.m uses 55 ps bins and 80 MHz.
timeBin_s = 55e-12;
repFreq_Hz = 80e6;

% MCX settings. Increase nphoton after the script is working.
nphoton = 1e6;
gpuid = 1;
useAutopilot = 1;
maxdetphoton = 2e6;

% Fit ranges. Units: mua 1/mm, D mm.
lowerBounds = [1e-5, 0.01];
upperBounds = [0.20, 100];
initialGuess = [0.005, 2.0];

% Optional timing shift between MCX+IRF and measured TCSPC curve.
fitTimeShiftBins = true;
timeShiftBounds_bins = [-60, 60];
initialTimeShift_bins = 0;

% If true, normalize target/IRF and fit only curve shape. The linear scale
% and constant background are still estimated for plotting.
useShapeOnlyResidual = true;

%% ---------------- Load data ----------------
add_mcx_paths(mcxlabPath);

targetData = load(targetFile);
irfData = load(irfFile);

targetHist = double(targetData.hist);
irfHist = double(irfData.hist);

targetCurve = squeeze(targetHist(pixelYX(1), pixelYX(2), :));
irfCurve = squeeze(irfHist(pixelYX(1), pixelYX(2), :));

targetCurve = clean_curve(targetCurve);
irfCurve = clean_curve(irfCurve);

nBins = numel(targetCurve);
if numel(irfCurve) ~= nBins
    error('Target and IRF curves have different lengths: %d vs %d.', nBins, numel(irfCurve));
end

tAxis_s = ((0:nBins-1).' + 0.5) * timeBin_s;
binEdges_s = (0:nBins).' * timeBin_s;

fprintf('Loaded target: %s\n', targetFile);
fprintf('Loaded IRF:    %s\n', irfFile);
fprintf('Using pixel (y,x) = (%d,%d), %d bins, %.1f ps/bin.\n', ...
    pixelYX(1), pixelYX(2), nBins, timeBin_s * 1e12);

%% ---------------- Build base MCX config ----------------
baseCfg = make_base_mcx_cfg(sampleSize_mm, voxelSize_mm, nMedium, g, ...
    sourcePosition_mm, detectorPosition_mm, detectorRadius_mm, ...
    sourceAngle_deg, nphoton, gpuid, useAutopilot, maxdetphoton, ...
    binEdges_s(1), binEdges_s(end), timeBin_s);

%% ---------------- Run fit ----------------
if fitTimeShiftBins
    x0 = [log(initialGuess(:)); initialTimeShift_bins];
else
    x0 = log(initialGuess(:));
end

objective = @(x) objective_mcx_fit(x, baseCfg, targetCurve, irfCurve, ...
    binEdges_s, g, lowerBounds, upperBounds, useShapeOnlyResidual, ...
    fitTimeShiftBins, timeShiftBounds_bins);

opts = optimset('Display', 'iter', 'MaxIter', 80, 'TolX', 1e-3, 'TolFun', 1e-3);
[xBest, fval] = fminsearch(objective, x0, opts);

[muaBest, DBest, shiftBest] = unpack_fit_vector(xBest, fitTimeShiftBins);
[simBest, simRawBest, fitScale, fitBackground] = simulate_curve_for_fit( ...
    baseCfg, muaBest, DBest, g, irfCurve, binEdges_s, fitTimeShiftBins, shiftBest, targetCurve);

muspBest = 1 / (3 * DBest) - muaBest;
musBest = muspBest / (1 - g);

fprintf('\nBest fit:\n');
fprintf('  mua  = %.6g 1/mm\n', muaBest);
fprintf('  D    = %.6g mm\n', DBest);
fprintf('  musp = %.6g 1/mm\n', muspBest);
fprintf('  mus  = %.6g 1/mm  (g = %.3f)\n', musBest, g);
fprintf('  shift = %.3f bins (%.3f ns)\n', shiftBest, shiftBest * timeBin_s * 1e9);
fprintf('  fitted scale = %.6g, background = %.6g\n', fitScale, fitBackground);
fprintf('  objective = %.6g\n', fval);

fitResult = struct();
fitResult.mua_1_per_mm = muaBest;
fitResult.D_mm = DBest;
fitResult.musp_1_per_mm = muspBest;
fitResult.mus_1_per_mm = musBest;
fitResult.g = g;
fitResult.n = nMedium;
fitResult.timeShift_bins = shiftBest;
fitResult.timeBin_s = timeBin_s;
fitResult.scale = fitScale;
fitResult.background = fitBackground;
fitResult.objective = fval;
fitResult.targetCurve = targetCurve;
fitResult.irfCurve = irfCurve;
fitResult.simRawCurve = simRawBest;
fitResult.simConvolvedCurve = simBest;
fitResult.tAxis_s = tAxis_s;
fitResult.baseCfg = baseCfg;

save('fit_mcx_absorption_diffusion_result.mat', 'fitResult');

%% ---------------- Plot ----------------
figure('Color', 'w', 'Name', 'MCX absorption/diffusion fit');

subplot(2,1,1);
plot(tAxis_s * 1e9, targetCurve, 'k-', 'LineWidth', 1.2); hold on;
plot(tAxis_s * 1e9, simBest, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Time (ns)');
ylabel('Counts');
legend('Measured target', 'MCX + IRF fit', 'Location', 'best');
title(sprintf('mua = %.4g 1/mm, D = %.4g mm, shift = %.2f bins', muaBest, DBest, shiftBest));

subplot(2,1,2);
plot(tAxis_s * 1e9, normalize_area(targetCurve), 'k-', 'LineWidth', 1.2); hold on;
plot(tAxis_s * 1e9, normalize_area(simBest), 'r-', 'LineWidth', 1.5);
plot(tAxis_s * 1e9, normalize_area(irfCurve), 'b--', 'LineWidth', 1.0);
grid on;
xlabel('Time (ns)');
ylabel('Area-normalized signal');
legend('Measured target', 'MCX + IRF fit', 'IRF', 'Location', 'best');

%% ======================== Local functions ========================

function add_mcx_paths(mcxlabPath)
    addpath(genpath(mcxlabPath));

    % Some MCXStudio installations keep mcxdetphoton.m outside the MATLAB
    % mcxlab folder. Add the standard MCXSuite utility path when present.
    mcxStudioPath = fullfile(mcxlabPath, '..', '..');
    mcxUtilsPath = fullfile(mcxStudioPath, 'MCXSuite', 'mcx', 'utils');
    if exist(fullfile(mcxUtilsPath, 'mcxdetphoton.m'), 'file')
        addpath(mcxUtilsPath);
    end

    if exist('mcxlab', 'file') ~= 3 && exist('mcxlab', 'file') ~= 2
        error('mcxlab was not found after adding path: %s', mcxlabPath);
    end

    if exist('mcxdetphoton', 'file') ~= 2
        warning('fit_mcx:missingMcxdetphoton', ...
            ['mcxdetphoton.m was not found. If MCX detects photons but MATLAB fails ', ...
            'while returning detp, add the folder containing mcxdetphoton.m to the MATLAB path.']);
    end
end

function cfg = make_base_mcx_cfg(sampleSize_mm, voxelSize_mm, nMedium, g, ...
    sourcePosition_mm, detectorPosition_mm, detectorRadius_mm, ...
    sourceAngle_deg, nphoton, gpuid, useAutopilot, maxdetphoton, ...
    tstart, tend, tstep)

    volSize = round(sampleSize_mm ./ voxelSize_mm);
    if any(abs(volSize .* voxelSize_mm - sampleSize_mm) > 1e-9)
        error('sampleSize_mm must be divisible by voxelSize_mm.');
    end

    theta = deg2rad(sourceAngle_deg);

    cfg = struct();
    cfg.nphoton = nphoton;
    cfg.vol = uint8(ones(volSize(1), volSize(2), volSize(3)));
    cfg.unitinmm = voxelSize_mm;
    cfg.srcpos = sourcePosition_mm ./ voxelSize_mm;
    cfg.srcdir = [sin(theta), 0, cos(theta)];
    cfg.detpos = [detectorPosition_mm ./ voxelSize_mm, detectorRadius_mm ./ voxelSize_mm];
    cfg.gpuid = gpuid;
    cfg.autopilot = useAutopilot;
    cfg.isreflect = 0;
    cfg.bc = 'aaaaaa';
    cfg.maxdetphoton = maxdetphoton;
    cfg.tstart = tstart;
    cfg.tend = tend;
    cfg.tstep = tstep;
    cfg.seed = 1648335518;

    % Placeholder. The objective overwrites row 2 for every parameter set.
    cfg.prop = [0, 0, 1, 1; 0.005, 1, g, nMedium];
end

function err = objective_mcx_fit(x, baseCfg, targetCurve, irfCurve, binEdges_s, ...
    g, lowerBounds, upperBounds, useShapeOnlyResidual, fitTimeShiftBins, timeShiftBounds_bins)

    [mua, D, shiftBins] = unpack_fit_vector(x, fitTimeShiftBins);

    if mua < lowerBounds(1) || mua > upperBounds(1) || ...
            D < lowerBounds(2) || D > upperBounds(2)
        err = 1e12 + bounds_penalty([mua, D], lowerBounds, upperBounds);
        return;
    end

    if fitTimeShiftBins && (shiftBins < timeShiftBounds_bins(1) || shiftBins > timeShiftBounds_bins(2))
        err = 1e12 + 1e6 * min(abs(shiftBins - timeShiftBounds_bins));
        return;
    end

    musp = 1 / (3 * D) - mua;
    if musp <= 0 || ~isfinite(musp)
        err = 1e12 + abs(musp) * 1e6;
        return;
    end

    try
        [modelCurve, ~] = simulate_curve_for_fit(baseCfg, mua, D, g, irfCurve, ...
            binEdges_s, fitTimeShiftBins, shiftBins, targetCurve);
    catch ME
        warning(ME.identifier, 'MCX evaluation failed: %s', ME.message);
        err = 1e12;
        return;
    end

    if useShapeOnlyResidual
        y = normalize_area(targetCurve);
        m = normalize_area(modelCurve);
        residual = (m - y) ./ sqrt(max(y, max(y) * 0.01) + eps);
    else
        residual = (modelCurve - targetCurve) ./ sqrt(max(targetCurve, 1));
    end

    err = sum(residual.^2);
    if ~isfinite(err)
        err = 1e12;
    end
end

function [modelCurve, rawCurve, scale, background] = simulate_curve_for_fit( ...
    baseCfg, mua, D, g, irfCurve, binEdges_s, fitTimeShiftBins, shiftBins, targetCurve)

    musp = 1 / (3 * D) - mua;
    mus = musp / (1 - g);

    cfg = baseCfg;
    cfg.prop(2, :) = [mua, mus, g, baseCfg.prop(2, 4)];

    [~, detp] = mcxlab(cfg);
    rawCurve = detphoton_to_tof_curve(detp, cfg, binEdges_s);

    if sum(rawCurve) <= 0
        error('No detected photons. Increase nphoton, detectorRadius_mm, or check geometry.');
    end

    convCurve = conv(normalize_area(rawCurve), normalize_area(irfCurve), 'same');
    convCurve = normalize_area(convCurve);

    if fitTimeShiftBins
        convCurve = shift_curve_fractional(convCurve, shiftBins);
    end

    A = [convCurve(:), ones(numel(convCurve), 1)];
    coeff = lsqnonneg(A, targetCurve(:));
    scale = coeff(1);
    background = coeff(2);
    modelCurve = scale * convCurve(:) + background;
end

function curve = detphoton_to_tof_curve(detp, cfg, binEdges_s)
    nBins = numel(binEdges_s) - 1;
    curve = zeros(nBins, 1);

    if isempty(detp) || ~isfield(detp, 'ppath') || isempty(detp.ppath)
        return;
    end

    ppath_mm = double(detp.ppath);
    nList = cfg.prop(2:end, 4);
    nList = nList(:).';

    if size(ppath_mm, 2) > numel(nList)
        nList(end+1:size(ppath_mm, 2)) = nList(end);
    end

    c0_mm_s = 299792458 * 1000;
    tof_s = sum(ppath_mm .* nList(1:size(ppath_mm, 2)), 2) ./ c0_mm_s;

    weights = ones(size(tof_s));
    if isfield(detp, 'w0') && numel(detp.w0) == numel(tof_s)
        weights = double(detp.w0(:));
    end

    curve = histcounts(tof_s, binEdges_s, 'Weights', weights).';
end

function [mua, D, shiftBins] = unpack_fit_vector(x, fitTimeShiftBins)
    mua = exp(x(1));
    D = exp(x(2));
    if fitTimeShiftBins
        shiftBins = x(3);
    else
        shiftBins = 0;
    end
end

function y = clean_curve(y)
    y = double(y(:));
    y(~isfinite(y)) = 0;
    y(y < 0) = 0;
end

function y = normalize_area(y)
    y = clean_curve(y);
    s = sum(y);
    if s > 0
        y = y ./ s;
    end
end

function yShift = shift_curve_fractional(y, shiftBins)
    y = y(:);
    x = (1:numel(y)).';
    yShift = interp1(x, y, x - shiftBins, 'linear', 0);
end

function p = bounds_penalty(value, lowerBounds, upperBounds)
    below = max(lowerBounds - value, 0);
    above = max(value - upperBounds, 0);
    p = 1e6 * sum(below.^2 + above.^2);
end
