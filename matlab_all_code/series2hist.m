function [histCube, timeAxis_ns, binEdges_ns] = series2hist(series, repFreq, varargin)
%SERIES2HIST Convert 32 x 32 x N photon arrival series to histogram cube.
%
% Usage:
%   [histCube, timeAxis_ns] = series2hist(series, 80e6);
%   [histCube, timeAxis_ns] = series2hist(series, 80);
%   [histCube, timeAxis_ns] = series2hist(series, 80e6, 'TimeBin', 55e-12);
%
% Input:
%   series:
%       32 x 32 x N matrix.
%       series(y, x, k) records the raw TDC code or raw arrival time
%       of the k-th photon at pixel (x, y).
%
%   repFreq:
%       Laser repetition frequency.
%       Unit: Hz.
%       Example:
%           80e6 means 80 MHz.
%
%       Convenience:
%           If repFreq < 1e5, it is automatically interpreted as MHz.
%           Therefore, repFreq = 80 is treated as 80 MHz.
%
% Name-value options:
%   'TimeBin':
%       TDC time bin width in seconds.
%       Default: 55e-12, namely 55 ps.
%
%   'SeriesUnit':
%       'bin' : series stores TDC bin code. rawTime = series * TimeBin.
%       's'   : series already stores time in seconds.
%       'ns'  : series stores time in nanoseconds.
%       'ps'  : series stores time in picoseconds.
%       Default: 'bin'.
%
%   'ReverseTCSPC':
%       true:
%           reversed TCSPC mode.
%           START = detected photon.
%           STOP  = laser sync.
%           correctedTime = T_rep - rawTime.
%
%       false:
%           normal TCSPC mode.
%           correctedTime = rawTime.
%
%       Default: true.
%
%   'ZeroIsPadding':
%       true:
%           series value 0 is treated as invalid padding.
%
%       false:
%           series value 0 is treated as a valid TDC value.
%
%       Default: true.
%
% Output:
%   histCube:
%       32 x 32 x M histogram cube.
%       M is determined by laser repetition period and TimeBin.
%       For example:
%           repFreq = 80 MHz
%           T_rep = 12.5 ns
%           TimeBin = 55 ps
%           M approximately equals ceil(12.5 ns / 55 ps) = 228.
%
%   timeAxis_ns:
%       1 x M vector.
%       Histogram bin centers in ns.
%
%   binEdges_ns:
%       1 x (M + 1) vector.
%       Histogram bin edges in ns.

    %% ---------------- Parse input ----------------
    if nargin < 2
        error('Usage: series2hist(series, repFreq, ...). repFreq is required.');
    end

    p = inputParser;

    addParameter(p, 'TimeBin', 55e-12, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'SeriesUnit', 'bin', @(x) ischar(x) || isstring(x));
    addParameter(p, 'ReverseTCSPC', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'ZeroIsPadding', true, @(x) islogical(x) || isnumeric(x));

    parse(p, varargin{:});

    timeBin_s = p.Results.TimeBin;
    seriesUnit = lower(string(p.Results.SeriesUnit));
    reverseTCSPC = logical(p.Results.ReverseTCSPC);
    zeroIsPadding = logical(p.Results.ZeroIsPadding);

    %% ---------------- Check series ----------------
    if ndims(series) ~= 3
        error('series must be a 3D matrix with size 32 x 32 x N.');
    end

    [ny, nx, ~] = size(series);

    if ny ~= 32 || nx ~= 32
        error('The first two dimensions of series must be 32 x 32.');
    end

    series = double(series);

    %% ---------------- Check repetition frequency ----------------
    if ~isnumeric(repFreq) || ~isscalar(repFreq) || repFreq <= 0
        error('repFreq must be a positive scalar.');
    end

    % If user enters 80, interpret it as 80 MHz.
    if repFreq < 1e5
        repFreq = repFreq * 1e6;
    end

    T_rep_s = 1 / repFreq;

    if timeBin_s >= T_rep_s
        error('TimeBin must be smaller than the laser repetition period.');
    end

    %% ---------------- Build histogram time axis ----------------
    % Number of bins is determined by repetition period and time bin.
    % It is not fixed to 1024.
    nBins = ceil(T_rep_s / timeBin_s);

    binEdges_s = (0:nBins) * timeBin_s;

    % Force the upper limit to be exactly the laser period.
    % The last bin may therefore be slightly shorter than TimeBin.
    binEdges_s(end) = T_rep_s;

    timeAxis_s = 0.5 * (binEdges_s(1:end-1) + binEdges_s(2:end));

    timeAxis_ns = timeAxis_s * 1e9;
    binEdges_ns = binEdges_s * 1e9;

    %% ---------------- Convert series to histogram cube ----------------
    histCube = zeros(ny, nx, nBins);

    for iy = 1:ny
        for ix = 1:nx

            values = squeeze(series(iy, ix, :));
            values = values(:);

            % Remove NaN and Inf
            values = values(isfinite(values));

            % Remove padding if needed
            if zeroIsPadding
                values = values(values > 0);
            else
                values = values(values >= 0);
            end

            if isempty(values)
                continue;
            end

            %% -------- Convert raw series values to raw time --------
            switch seriesUnit
                case "bin"
                    rawTime_s = values * timeBin_s;

                case "s"
                    rawTime_s = values;

                case "ns"
                    rawTime_s = values * 1e-9;

                case "ps"
                    rawTime_s = values * 1e-12;

                otherwise
                    error('Unsupported SeriesUnit. Use "bin", "s", "ns", or "ps".');
            end

            %% -------- Keep only raw values within one pulse period --------
            rawTime_s = rawTime_s(rawTime_s >= 0 & rawTime_s <= T_rep_s);

            if isempty(rawTime_s)
                continue;
            end

            %% -------- Correct reversed TCSPC time axis --------
            if reverseTCSPC
                correctedTime_s = T_rep_s - rawTime_s;
            else
                correctedTime_s = rawTime_s;
            end

            % Numerical protection
            correctedTime_s(correctedTime_s < 0) = 0;
            correctedTime_s(correctedTime_s > T_rep_s) = T_rep_s;

            %% -------- Histogram --------
            counts = histcounts(correctedTime_s, binEdges_s);

            histCube(iy, ix, :) = reshape(counts, 1, 1, nBins);
        end
    end
end