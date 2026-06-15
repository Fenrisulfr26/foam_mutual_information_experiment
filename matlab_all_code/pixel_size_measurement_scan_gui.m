function pixel_size_measurement_scan_gui()
%PIXEL_SIZE_MEASUREMENT_SCAN_GUI Galvo voltage grid scan with SPAD capture.
%
% Row-major scan order:
%   X: left to right, 0.06 V -> -0.18 V
%   Y: top to bottom, 0.15 V -> -0.10 V
%
% Default grid:
%   25 x 25 points.
%   X step is exactly 0.01 V.
%   Y is linspace(0.15, -0.10, 25), step ~= 0.0104167 V.
%
% Output:
%   data/Pixel size measurement/<run_folder>/
%       hist_pointXXXX_rowYY_colXX.mat
%       scan_log.mat
%       scan_metadata.mat

%% Paths / defaults
PROJECT_ROOT = fileparts(mfilename('fullpath'));
addpath(genpath(PROJECT_ROOT));

PF_WRAPPER_PATHS = {
    'D:\coding\Photon Force\PF32\Matlab\my_wrapper'
    'D:\coding\PF32\Matlab'
};

for pathIdx = 1:numel(PF_WRAPPER_PATHS)
    if isfolder(PF_WRAPPER_PATHS{pathIdx})
        addpath(genpath(PF_WRAPPER_PATHS{pathIdx}));
    end
end

DEFAULT_AFG_VISA_ADDRESS = 'USB0::0x0699::0x035E::C018251::INSTR';
DEFAULT_EXPO_TIME_US = 2;
DEFAULT_NUM_FRAMES = 1e5;
DEFAULT_HIST_REPEATS = 1;
DEFAULT_LASER_REP_FREQ_HZ = 80.33e6;
DEFAULT_GALVO_SETTLE_S = 0.2;
DEFAULT_GRID_POINTS = 25;
DEFAULT_X_START_V = 0.06;
DEFAULT_X_END_V = -0.18;
DEFAULT_Y_START_V = 0.15;
DEFAULT_Y_END_V = -0.10;
DEFAULT_DISPLAY_EACH_HIST = false;
DEFAULT_ALLOW_OVERWRITE = false;
PARK_X_V = -0.060;
PARK_Y_V = 0.025;

state = struct();
state.afg = [];
state.pf32 = [];
state.isGalvoOpen = false;
state.isSpadOpen = false;
state.isBusy = false;
state.outputDir = '';
state.runFolderName = '';

handles = buildUi();
refreshHardwareStatus();
refreshOutputFolderLabel();
logMessage('UI opened. Initialize hardware, check parameters, then start scan.');

%% UI
    function handles = buildUi()
        handles.fig = figure( ...
            'Name', 'Pixel Size Measurement Scan', ...
            'NumberTitle', 'off', ...
            'MenuBar', 'none', ...
            'ToolBar', 'none', ...
            'Resize', 'off', ...
            'Units', 'pixels', ...
            'Position', [140, 70, 1080, 820], ...
            'CloseRequestFcn', @onCloseFigure);

        handles.title = uicontrol(handles.fig, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [20, 782, 480, 24], ...
            'HorizontalAlignment', 'left', ...
            'FontSize', 13, ...
            'FontWeight', 'bold', ...
            'String', 'Pixel Size Measurement Scan');

        handles.status = uicontrol(handles.fig, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [540, 782, 520, 24], ...
            'HorizontalAlignment', 'right', ...
            'FontSize', 10, ...
            'String', 'Hardware: not initialized');

        handles.paramPanel = uipanel(handles.fig, ...
            'Units', 'pixels', ...
            'Title', 'Parameters', ...
            'Position', [20, 330, 430, 435]);

        xLabel = 18;
        xEdit = 180;
        y0 = 382;
        dy = 31;
        editW = 215;

        addLabel(handles.paramPanel, xLabel, y0, 'AFG VISA');
        handles.afgVisaEdit = addEdit(handles.paramPanel, xEdit, y0, editW, DEFAULT_AFG_VISA_ADDRESS);

        addLabel(handles.paramPanel, xLabel, y0 - dy, 'Exposure us');
        handles.expoEdit = addEdit(handles.paramPanel, xEdit, y0 - dy, editW, num2str(DEFAULT_EXPO_TIME_US));

        addLabel(handles.paramPanel, xLabel, y0 - 2*dy, 'Frames per hist');
        handles.framesEdit = addEdit(handles.paramPanel, xEdit, y0 - 2*dy, editW, num2str(DEFAULT_NUM_FRAMES));

        addLabel(handles.paramPanel, xLabel, y0 - 3*dy, 'Hist repeats per point');
        handles.repeatsEdit = addEdit(handles.paramPanel, xEdit, y0 - 3*dy, editW, num2str(DEFAULT_HIST_REPEATS));

        addLabel(handles.paramPanel, xLabel, y0 - 4*dy, 'Laser rep freq Hz');
        handles.freqEdit = addEdit(handles.paramPanel, xEdit, y0 - 4*dy, editW, num2str(DEFAULT_LASER_REP_FREQ_HZ));

        addLabel(handles.paramPanel, xLabel, y0 - 5*dy, 'Galvo settle s');
        handles.settleEdit = addEdit(handles.paramPanel, xEdit, y0 - 5*dy, editW, num2str(DEFAULT_GALVO_SETTLE_S));

        addLabel(handles.paramPanel, xLabel, y0 - 6*dy, 'Grid points');
        handles.gridPointsEdit = addEdit(handles.paramPanel, xEdit, y0 - 6*dy, editW, num2str(DEFAULT_GRID_POINTS));

        addLabel(handles.paramPanel, xLabel, y0 - 7*dy, 'X start / end V');
        handles.xStartEdit = addEdit(handles.paramPanel, xEdit, y0 - 7*dy, 100, num2str(DEFAULT_X_START_V));
        handles.xEndEdit = addEdit(handles.paramPanel, xEdit + 115, y0 - 7*dy, 100, num2str(DEFAULT_X_END_V));

        addLabel(handles.paramPanel, xLabel, y0 - 8*dy, 'Y start / end V');
        handles.yStartEdit = addEdit(handles.paramPanel, xEdit, y0 - 8*dy, 100, num2str(DEFAULT_Y_START_V));
        handles.yEndEdit = addEdit(handles.paramPanel, xEdit + 115, y0 - 8*dy, 100, num2str(DEFAULT_Y_END_V));

        handles.displayHistCheck = uicontrol(handles.paramPanel, 'Style', 'checkbox', ...
            'Units', 'pixels', ...
            'Position', [18, 74, 160, 22], ...
            'Value', DEFAULT_DISPLAY_EACH_HIST, ...
            'String', 'Display each hist');

        handles.overwriteCheck = uicontrol(handles.paramPanel, 'Style', 'checkbox', ...
            'Units', 'pixels', ...
            'Position', [205, 74, 160, 22], ...
            'Value', DEFAULT_ALLOW_OVERWRITE, ...
            'String', 'Allow overwrite');

        addLabel(handles.paramPanel, xLabel, 45, 'Note');
        handles.noteEdit = uicontrol(handles.paramPanel, 'Style', 'edit', ...
            'Units', 'pixels', ...
            'Position', [18, 10, 385, 34], ...
            'HorizontalAlignment', 'left', ...
            'Max', 2, ...
            'Min', 0, ...
            'String', '');

        handles.controlPanel = uipanel(handles.fig, ...
            'Units', 'pixels', ...
            'Title', 'Control', ...
            'Position', [470, 540, 590, 225]);

        handles.initButton = addButton(handles.controlPanel, [25, 160, 160, 36], ...
            'Initialize hardware', @onInitHardware);
        handles.closeButton = addButton(handles.controlPanel, [205, 160, 160, 36], ...
            'Close hardware', @onCloseHardware);
        handles.newFolderButton = addButton(handles.controlPanel, [385, 160, 160, 36], ...
            'New output folder', @onNewOutputFolder);

        handles.startButton = addButton(handles.controlPanel, [25, 96, 520, 44], ...
            'Start pixel size scan', @onStartScan);

        handles.openFolderButton = addButton(handles.controlPanel, [25, 48, 250, 32], ...
            'Open output folder', @onOpenOutputFolder);
        handles.previewButton = addButton(handles.controlPanel, [295, 48, 250, 32], ...
            'Preview voltage grid', @onPreviewGrid);

        handles.folderText = uicontrol(handles.controlPanel, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [25, 8, 520, 28], ...
            'HorizontalAlignment', 'left', ...
            'String', '');

        handles.progressPanel = uipanel(handles.fig, ...
            'Units', 'pixels', ...
            'Title', 'Progress', ...
            'Position', [470, 330, 590, 190]);

        handles.progressText = uicontrol(handles.progressPanel, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [20, 120, 545, 35], ...
            'HorizontalAlignment', 'left', ...
            'String', 'Idle');

        handles.currentPointText = uicontrol(handles.progressPanel, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [20, 80, 545, 28], ...
            'HorizontalAlignment', 'left', ...
            'String', '');

        handles.progressAxes = axes( ...
            'Parent', handles.progressPanel, ...
            'Units', 'pixels', ...
            'Position', [35, 25, 515, 38], ...
            'XLim', [0, 1], ...
            'YLim', [0, 1], ...
            'Box', 'on', ...
            'XTick', [], ...
            'YTick', []);
        handles.progressPatch = patch(handles.progressAxes, ...
            [0 0 0 0], [0 1 1 0], [0.2 0.55 0.85], ...
            'EdgeColor', 'none');

        handles.logBox = uicontrol(handles.fig, 'Style', 'listbox', ...
            'Units', 'pixels', ...
            'Position', [20, 20, 1040, 290], ...
            'FontName', 'Consolas', ...
            'FontSize', 9, ...
            'String', {});
    end

%% callbacks
    function onInitHardware(~, ~)
        if state.isBusy
            return;
        end

        setBusy(true);
        try
            cfg = readConfigFromUi();
            logMessage('Initializing AFG and SPAD...');

            closeHardwareQuietly();

            state.afg = openAfg(cfg.afgVisaAddress);
            state.isGalvoOpen = true;

            state.pf32 = openSpad();
            state.isSpadOpen = true;
            configureSpad(state.pf32, cfg.expoTimeUs);

            refreshHardwareStatus();
            logMessage('Hardware initialized.');
        catch ME
            logError(ME);
            closeHardwareQuietly();
        end
        setBusy(false);
    end

    function onCloseHardware(~, ~)
        if state.isBusy
            return;
        end

        closeHardwareQuietly();
        refreshHardwareStatus();
        logMessage('Hardware closed.');
    end

    function onNewOutputFolder(~, ~)
        if state.isBusy
            return;
        end

        state.outputDir = '';
        state.runFolderName = '';
        refreshOutputFolderLabel();
        resetProgress();
        logMessage('Next scan will create a new output folder.');
    end

    function onOpenOutputFolder(~, ~)
        try
            rootDir = getPixelSizeOutputRoot();
            if isempty(state.outputDir)
                targetDir = rootDir;
            else
                targetDir = state.outputDir;
            end
            if ~exist(targetDir, 'dir')
                mkdir(targetDir);
            end
            winopen(targetDir);
            logMessage(sprintf('Opened folder: %s', targetDir));
        catch ME
            logError(ME);
        end
    end

    function onPreviewGrid(~, ~)
        try
            cfg = readConfigFromUi();
            [xVoltages, yVoltages, scanTable] = makeVoltageGrid(cfg);

            fig = figure( ...
                'Name', 'Pixel size scan voltage grid', ...
                'NumberTitle', 'off', ...
                'Color', 'w', ...
                'Position', [220, 120, 780, 660]);
            ax = axes('Parent', fig);
            plot(ax, scanTable.x_v, scanTable.y_v, 'o-', 'LineWidth', 1.0);
            grid(ax, 'on');
            axis(ax, 'equal');
            xlabel(ax, 'X voltage (V)');
            ylabel(ax, 'Y voltage (V)');
            title(ax, sprintf('%d x %d row-major scan, %d points', ...
                numel(yVoltages), numel(xVoltages), height(scanTable)));
            set(ax, 'YDir', 'normal');

            logMessage(sprintf(['Grid preview: X %.6g -> %.6g V, step %.6g V; ', ...
                'Y %.6g -> %.6g V, step %.6g V; total %d points.'], ...
                xVoltages(1), xVoltages(end), xVoltages(2) - xVoltages(1), ...
                yVoltages(1), yVoltages(end), yVoltages(2) - yVoltages(1), ...
                height(scanTable)));
        catch ME
            logError(ME);
        end
    end

    function onStartScan(~, ~)
        if state.isBusy
            return;
        end

        setBusy(true);
        try
            ensureHardwareReady();
            cfg = readConfigFromUi();
            configureSpad(state.pf32, cfg.expoTimeUs);

            [xVoltages, yVoltages, scanTable] = makeVoltageGrid(cfg);
            outputDir = prepareOutputFolder(cfg, xVoltages, yVoltages, scanTable);

            logMessage(sprintf('Starting scan: %d x %d = %d points.', ...
                numel(yVoltages), numel(xVoltages), height(scanTable)));
            logMessage(sprintf('Output folder: %s', outputDir));

            scanLog = repmat(blankScanInfo(), height(scanTable), 1);

            totalTimer = tic;
            for idx = 1:height(scanTable)
                rowIdx = scanTable.row(idx);
                colIdx = scanTable.col(idx);
                pointIdx = scanTable.point_index(idx);
                targetX = scanTable.x_v(idx);
                targetY = scanTable.y_v(idx);

                updateProgress(idx - 1, height(scanTable), ...
                    sprintf('Moving to point %d/%d, row %d, col %d', ...
                    idx, height(scanTable), rowIdx, colIdx));

                writeGalvoVoltage(state.afg, targetX, targetY);
                if cfg.galvoSettleS > 0
                    pause(cfg.galvoSettleS);
                end

                logMessage(sprintf('Point %d/%d: row=%02d col=%02d X=%.6f V Y=%.6f V', ...
                    idx, height(scanTable), rowIdx, colIdx, targetX, targetY));

                tic;
                [hist, timeAxis_ns, binEdges_ns, repeatElapsedS] = acquireAverageHist( ...
                    state.pf32, cfg.numFrames, cfg.laserRepFreqHz, cfg.histRepeats, idx);
                elapsedS = toc;

                scanInfo = blankScanInfo();
                scanInfo.point_index = pointIdx;
                scanInfo.row = rowIdx;
                scanInfo.col = colIdx;
                scanInfo.target_voltage_x_v = targetX;
                scanInfo.target_voltage_y_v = targetY;
                scanInfo.expo_time_us = cfg.expoTimeUs;
                scanInfo.num_frames = cfg.numFrames;
                scanInfo.hist_repeats = cfg.histRepeats;
                scanInfo.laser_rep_freq_hz = cfg.laserRepFreqHz;
                scanInfo.galvo_settle_s = cfg.galvoSettleS;
                scanInfo.elapsed_s = elapsedS;
                scanInfo.repeat_elapsed_s = repeatElapsedS;
                scanInfo.acquired_at = timestampText();

                fileName = sprintf('hist_point%04d_row%02d_col%02d_x_%s_y_%s.mat', ...
                    pointIdx, rowIdx, colIdx, voltageToken(targetX), voltageToken(targetY));
                filePath = fullfile(outputDir, fileName);
                scanInfo.file = char(filePath);

                if exist(filePath, 'file') && ~cfg.allowOverwrite
                    error('Output file already exists: %s', filePath);
                end

                save(filePath, 'hist', 'timeAxis_ns', 'binEdges_ns', 'scanInfo');

                if cfg.displayEachHist
                    my_display_hist(hist);
                    drawnow;
                end

                scanLog(idx, 1) = scanInfo;
                clear hist;

                updateProgress(idx, height(scanTable), ...
                    sprintf('Saved point %d/%d, elapsed %.2f s', idx, height(scanTable), elapsedS));
            end

            totalElapsedS = toc(totalTimer);
            save(fullfile(outputDir, 'scan_log.mat'), 'scanLog');
            updateScanMetadataFinished(outputDir, totalElapsedS);

            updateProgress(height(scanTable), height(scanTable), ...
                sprintf('Finished %d points, total %.2f s', height(scanTable), totalElapsedS));
            logMessage(sprintf('Scan finished. Total elapsed %.2f s.', totalElapsedS));
        catch ME
            logError(ME);
        end

        setBusy(false);
    end

    function onCloseFigure(~, ~)
        if state.isBusy
            choice = questdlg('Scan is running. Close UI anyway?', ...
                'Close confirmation', 'Close', 'Cancel', 'Cancel');
            if ~strcmp(choice, 'Close')
                return;
            end
        end

        closeHardwareQuietly();
        delete(handles.fig);
    end

%% scan helpers
    function [xVoltages, yVoltages, scanTable] = makeVoltageGrid(cfg)
        n = cfg.gridPoints;
        xVoltages = linspace(cfg.xStartV, cfg.xEndV, n);
        yVoltages = linspace(cfg.yStartV, cfg.yEndV, n);

        validateVoltageVector(xVoltages, 'X');
        validateVoltageVector(yVoltages, 'Y');

        pointIndex = zeros(n * n, 1);
        row = zeros(n * n, 1);
        col = zeros(n * n, 1);
        x = zeros(n * n, 1);
        y = zeros(n * n, 1);

        idx = 0;
        for r = 1:n
            for c = 1:n
                idx = idx + 1;
                pointIndex(idx) = idx;
                row(idx) = r;
                col(idx) = c;
                x(idx) = xVoltages(c);
                y(idx) = yVoltages(r);
            end
        end

        scanTable = table(pointIndex, row, col, x, y, ...
            'VariableNames', {'point_index', 'row', 'col', 'x_v', 'y_v'});
    end

    function outputDir = prepareOutputFolder(cfg, xVoltages, yVoltages, scanTable)
        rootDir = getPixelSizeOutputRoot();
        if ~exist(rootDir, 'dir')
            mkdir(rootDir);
        end

        if isempty(state.outputDir)
            stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
            state.runFolderName = sprintf('pixel_size_measurement_%s_%dx%d', ...
                stamp, cfg.gridPoints, cfg.gridPoints);
            state.outputDir = fullfile(rootDir, state.runFolderName);
        end

        outputDir = state.outputDir;
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end

        metadata = struct();
        metadata.created_at = timestampText();
        metadata.project_root = char(PROJECT_ROOT);
        metadata.output_dir = char(outputDir);
        metadata.run_folder_name = state.runFolderName;
        metadata.afg_visa_address = cfg.afgVisaAddress;
        metadata.afg_output_load = 'High-Z / MAX';
        metadata.park_voltage_v = [PARK_X_V, PARK_Y_V];
        metadata.expo_time_us = cfg.expoTimeUs;
        metadata.num_frames = cfg.numFrames;
        metadata.hist_repeats = cfg.histRepeats;
        metadata.laser_rep_freq_hz = cfg.laserRepFreqHz;
        metadata.galvo_settle_s = cfg.galvoSettleS;
        metadata.grid_points = cfg.gridPoints;
        metadata.x_voltages_v = xVoltages;
        metadata.y_voltages_v = yVoltages;
        metadata.x_step_v = xVoltages(2) - xVoltages(1);
        metadata.y_step_v = yVoltages(2) - yVoltages(1);
        metadata.scan_order = 'row-major: left-to-right, then top-to-bottom';
        metadata.scan_table = scanTable;
        metadata.experiment_note = cfg.experimentNote;

        metadataPath = fullfile(outputDir, 'scan_metadata.mat');
        logPath = fullfile(outputDir, 'scan_log.mat');
        if exist(logPath, 'file') && ~cfg.allowOverwrite
            error('Output folder already contains scan_log.mat: %s', outputDir);
        end

        save(metadataPath, 'metadata');
        refreshOutputFolderLabel();
    end

    function updateScanMetadataFinished(outputDir, totalElapsedS)
        metadataPath = fullfile(outputDir, 'scan_metadata.mat');
        if exist(metadataPath, 'file')
            data = load(metadataPath, 'metadata');
            metadata = data.metadata;
            metadata.finished_at = timestampText();
            metadata.total_elapsed_s = totalElapsedS;
            save(metadataPath, 'metadata');
        end
    end

    function [histAvg, timeAxis_ns, binEdges_ns, repeatElapsedS] = acquireAverageHist( ...
            pf32, numFrames, laserRepFreqHz, histRepeats, pointIdx)

        histSum = [];
        timeAxis_ns = [];
        binEdges_ns = [];
        repeatElapsedS = zeros(histRepeats, 1);

        for repIdx = 1:histRepeats
            logMessage(sprintf('Point %d: hist repeat %d/%d, frames %.0f', ...
                pointIdx, repIdx, histRepeats, numFrames));

            [histOne, timeAxis_ns, binEdges_ns, repeatElapsedS(repIdx)] = acquireSingleHist( ...
                pf32, numFrames, laserRepFreqHz);

            if isempty(histSum)
                histSum = double(histOne);
            else
                histSum = histSum + double(histOne);
            end

            clear histOne;
        end

        histAvg = histSum ./ histRepeats;
    end

    function [histOne, timeAxis_ns, binEdges_ns, elapsedS] = acquireSingleHist( ...
            pf32, numFrames, laserRepFreqHz)

        tic;
        series = pf_getMultipleFrames(pf32, numFrames);
        elapsedS = toc;

        [histOne, timeAxis_ns, binEdges_ns] = series2hist(series, laserRepFreqHz);
        histOne = double(histOne);
        clear series;
    end

%% config / state
    function cfg = readConfigFromUi()
        cfg = struct();
        cfg.afgVisaAddress = strtrim(get(handles.afgVisaEdit, 'String'));
        cfg.expoTimeUs = readPositiveNumber(handles.expoEdit, 'Exposure us');
        cfg.numFrames = round(readPositiveNumber(handles.framesEdit, 'Frames per hist'));
        cfg.histRepeats = round(readPositiveNumber(handles.repeatsEdit, 'Hist repeats per point'));
        cfg.laserRepFreqHz = readPositiveNumber(handles.freqEdit, 'Laser rep freq Hz');
        cfg.galvoSettleS = readNonnegativeNumber(handles.settleEdit, 'Galvo settle s');
        cfg.gridPoints = round(readPositiveNumber(handles.gridPointsEdit, 'Grid points'));
        cfg.xStartV = readFiniteNumber(handles.xStartEdit, 'X start V');
        cfg.xEndV = readFiniteNumber(handles.xEndEdit, 'X end V');
        cfg.yStartV = readFiniteNumber(handles.yStartEdit, 'Y start V');
        cfg.yEndV = readFiniteNumber(handles.yEndEdit, 'Y end V');
        cfg.displayEachHist = logical(get(handles.displayHistCheck, 'Value'));
        cfg.allowOverwrite = logical(get(handles.overwriteCheck, 'Value'));
        cfg.experimentNote = readMultilineText(handles.noteEdit);

        if isempty(cfg.afgVisaAddress)
            error('AFG VISA address cannot be empty.');
        end
        if cfg.gridPoints < 2
            error('Grid points must be >= 2.');
        end
        if cfg.histRepeats < 1
            error('Hist repeats per point must be >= 1.');
        end
        if cfg.numFrames < 1
            error('Frames per hist must be >= 1.');
        end
        validateAfgVoltage(cfg.xStartV, 'X start');
        validateAfgVoltage(cfg.xEndV, 'X end');
        validateAfgVoltage(cfg.yStartV, 'Y start');
        validateAfgVoltage(cfg.yEndV, 'Y end');
    end

    function rootDir = getPixelSizeOutputRoot()
        rootDir = fullfile(PROJECT_ROOT, 'data', 'Pixel size measurement');
    end

    function setBusy(isBusy)
        state.isBusy = isBusy;
        controls = [
            handles.initButton
            handles.closeButton
            handles.newFolderButton
            handles.startButton
            handles.openFolderButton
            handles.previewButton
        ];

        if isBusy
            set(controls, 'Enable', 'off');
            set(handles.status, 'String', 'Hardware: busy');
        else
            set(controls, 'Enable', 'on');
            refreshHardwareStatus();
        end
        drawnow;
    end

    function refreshHardwareStatus()
        set(handles.status, 'String', sprintf('Hardware: AFG %s / SPAD %s', ...
            onOffText(state.isGalvoOpen), onOffText(state.isSpadOpen)));
    end

    function refreshOutputFolderLabel()
        if isempty(state.outputDir)
            set(handles.folderText, 'String', 'Output: auto-created under data\Pixel size measurement');
        else
            set(handles.folderText, 'String', sprintf('Output: %s', state.outputDir));
        end
    end

    function updateProgress(doneCount, totalCount, message)
        frac = 0;
        if totalCount > 0
            frac = min(max(doneCount / totalCount, 0), 1);
        end

        set(handles.progressPatch, 'XData', [0 frac frac 0], 'YData', [0 0 1 1]);
        set(handles.progressText, 'String', sprintf('Progress: %d / %d (%.1f%%)', ...
            doneCount, totalCount, 100 * frac));
        set(handles.currentPointText, 'String', message);
        drawnow;
    end

    function resetProgress()
        updateProgress(0, 1, 'Idle');
    end

    function ensureHardwareReady()
        if ~state.isGalvoOpen || isempty(state.afg)
            error('AFG/Galvo is not initialized.');
        end
        if ~state.isSpadOpen || isempty(state.pf32)
            error('SPAD is not initialized.');
        end
    end

    function closeHardwareQuietly()
        closeAfgState();
        closeSpadState();
    end

    function closeAfgState()
        if state.isGalvoOpen && ~isempty(state.afg)
            afgLocal = state.afg;
            parkAndReleaseAfg(afgLocal, PARK_X_V, PARK_Y_V);
            state.afg = [];
            state.isGalvoOpen = false;
        end
    end

    function closeSpadState()
        if state.isSpadOpen && ~isempty(state.pf32)
            pf32Local = state.pf32;
            state.pf32 = [];
            state.isSpadOpen = false;
            closeSpad(pf32Local);
        end
    end

%% logging
    function logMessage(message)
        timestamped = sprintf('[%s] %s', char(datetime('now', 'Format', 'HH:mm:ss')), message);
        currentLog = get(handles.logBox, 'String');
        if ischar(currentLog)
            currentLog = cellstr(currentLog);
        end
        currentLog{end + 1} = timestamped;
        set(handles.logBox, 'String', currentLog, 'Value', numel(currentLog));
        fprintf('%s\n', timestamped);
        drawnow;
    end

    function logError(ME)
        logMessage(sprintf('ERROR: %s', ME.message));
        for stackIdx = 1:numel(ME.stack)
            logMessage(sprintf('  at %s line %d', ME.stack(stackIdx).name, ME.stack(stackIdx).line));
        end
    end
end

%% hardware helpers
function afg = openAfg(visaAddress)
    afg = visadev(visaAddress);
    configureTerminator(afg, "LF");
    afg.Timeout = 5;

    idn = strtrim(writeread(afg, '*IDN?'));
    fprintf('Connected to AFG: %s\n', idn);

    configureAfgDcChannel(afg, 1);
    configureAfgDcChannel(afg, 2);
    writeline(afg, 'OUTP1:STAT ON');
    writeline(afg, 'OUTP2:STAT ON');
end

function pf32 = openSpad()
    try
        ok = pf_getLibraryStatus;
        if ok
            pf_close;
        end
    catch statusErr
        warning('SPAD library status check failed before opening: %s', statusErr.message);
    end

    [pf32, ~, ~] = pf_open;
    pf_getLibraryStatus;
end

function configureSpad(pf32, expoTimeUs)
    pf_setMode(pf32, 'TCSPC_laser_master');
    pf_setNumberAccumulations(pf32, 1);
    pf_setExposure(pf32, expoTimeUs);
end

function configureAfgDcChannel(afg, chan)
    writeline(afg, sprintf('OUTP%d:IMP MAX', chan));
    writeline(afg, sprintf('SOUR%d:FUNC:SHAP DC', chan));
    writeline(afg, sprintf('SOUR%d:VOLT:UNIT VPP', chan));
    writeline(afg, sprintf('SOUR%d:VOLT:AMPL 0.001VPP', chan));
end

function writeGalvoVoltage(afg, targetVoltageX, targetVoltageY)
    validateAfgVoltage(targetVoltageX, 'X');
    validateAfgVoltage(targetVoltageY, 'Y');
    setAfgDcVoltage(afg, 1, targetVoltageX);
    setAfgDcVoltage(afg, 2, targetVoltageY);
end

function setAfgDcVoltage(afg, chan, voltageV)
    writeline(afg, sprintf('SOUR%d:VOLT:OFFS %.9gV', chan, voltageV));
end

function validateAfgVoltage(voltageV, axisName)
    maxAbsVoltageV = 2;
    if ~isfinite(voltageV) || abs(voltageV) > maxAbsVoltageV
        error('Galvo %s voltage %.6g V exceeds safety range +/- %.3f V.', ...
            axisName, voltageV, maxAbsVoltageV);
    end
end

function validateVoltageVector(voltages, axisName)
    for idx = 1:numel(voltages)
        validateAfgVoltage(voltages(idx), axisName);
    end
end

function parkAndReleaseAfg(afg, parkXVoltageV, parkYVoltageV)
    if isempty(afg)
        return;
    end

    try
        writeGalvoVoltage(afg, parkXVoltageV, parkYVoltageV);
        writeline(afg, 'OUTP1:STAT ON');
        writeline(afg, 'OUTP2:STAT ON');
        pause(0.2);
    catch closeErr
        warning('Failed to park AFG outputs before releasing connection: %s', closeErr.message);
    end

    clear afg;
    fprintf('AFG parked at X = %.3f V, Y = %.3f V. Outputs left ON; MATLAB connection released.\n', ...
        parkXVoltageV, parkYVoltageV);
end

function closeSpad(pf32)
    try
        pf_close(pf32);
    catch
        try
            pf_close;
        catch closeErr
            warning('SPAD close failed: %s', closeErr.message);
        end
    end
end

%% small helpers
function scanInfo = blankScanInfo()
    scanInfo = struct( ...
        'point_index', [], ...
        'row', [], ...
        'col', [], ...
        'target_voltage_x_v', [], ...
        'target_voltage_y_v', [], ...
        'expo_time_us', [], ...
        'num_frames', [], ...
        'hist_repeats', [], ...
        'laser_rep_freq_hz', [], ...
        'galvo_settle_s', [], ...
        'elapsed_s', [], ...
        'repeat_elapsed_s', [], ...
        'acquired_at', '', ...
        'file', '');
end

function text = timestampText()
    text = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
end

function token = voltageToken(value)
    if value < 0
        prefix = 'neg';
    else
        prefix = 'pos';
    end
    token = sprintf('%s%sV', prefix, strrep(sprintf('%.6f', abs(value)), '.', 'p'));
end

function text = readMultilineText(editHandle)
    rawValue = get(editHandle, 'String');
    if iscell(rawValue)
        text = strjoin(rawValue(:).', newline);
    elseif isstring(rawValue)
        text = strjoin(cellstr(rawValue(:)), newline);
    elseif ischar(rawValue) && size(rawValue, 1) > 1
        text = strjoin(cellstr(rawValue), newline);
    else
        text = char(rawValue);
    end

    text = strrep(text, sprintf('\r\n'), newline);
    text = strrep(text, sprintf('\r'), newline);
end

function value = readPositiveNumber(editHandle, label)
    value = str2double(strtrim(get(editHandle, 'String')));
    if ~isfinite(value) || value <= 0
        error('%s must be a positive number.', label);
    end
end

function value = readNonnegativeNumber(editHandle, label)
    value = str2double(strtrim(get(editHandle, 'String')));
    if ~isfinite(value) || value < 0
        error('%s must be a nonnegative number.', label);
    end
end

function value = readFiniteNumber(editHandle, label)
    value = str2double(strtrim(get(editHandle, 'String')));
    if ~isfinite(value)
        error('%s must be a finite number.', label);
    end
end

function text = onOffText(isOn)
    if isOn
        text = 'on';
    else
        text = 'off';
    end
end

function addLabel(parent, x, y, labelText)
    uicontrol(parent, 'Style', 'text', ...
        'Units', 'pixels', ...
        'Position', [x, y, 150, 22], ...
        'HorizontalAlignment', 'left', ...
        'String', labelText);
end

function handle = addEdit(parent, x, y, width, valueText)
    handle = uicontrol(parent, 'Style', 'edit', ...
        'Units', 'pixels', ...
        'Position', [x, y, width, 24], ...
        'HorizontalAlignment', 'left', ...
        'String', valueText);
end

function handle = addButton(parent, position, label, callback)
    handle = uicontrol(parent, 'Style', 'pushbutton', ...
        'Units', 'pixels', ...
        'Position', position, ...
        'FontSize', 10, ...
        'String', label, ...
        'Callback', callback);
end
