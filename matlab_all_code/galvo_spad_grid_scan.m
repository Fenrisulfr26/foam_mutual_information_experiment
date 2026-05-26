function galvo_spad_grid_scan()
%GALVO_SPAD_GRID_SCAN UI controller for 3x3 galvo + SPAD histogram scans.
%   Run this function, then use the buttons in the UI to initialize hardware,
%   choose OBJ/REF mode, set how many histograms to average at each point,
%   and start the scan.

%% --- Defaults ---
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
DEFAULT_LASER_DEGREE = 2;
DEFAULT_EXPO_TIME_US = 2;
DEFAULT_NUM_FRAMES = 1e5;
DEFAULT_HIST_REPEATS = 1;
DEFAULT_LASER_REP_FREQ_HZ = 80.33e6;
DEFAULT_GALVO_SETTLE_S = 0.2;
DEFAULT_DISPLAY_EACH_HIST = false;
DEFAULT_ALLOW_OVERWRITE = false;
V_LIMIT_V = 2;
PARK_X_V = -0.090;
PARK_Y_V = 0.025;

scan_point_names = {
    '左上'
    '上中'
    '右上'
    '左中'
    '中心'
    '右中'
    '左下'
    '下中'
    '右下'
};

scan_point_ids = {
    'left_top'
    'top_middle'
    'right_top'
    'left_middle'
    'center'
    'right_middle'
    'left_bottom'
    'bottom_middle'
    'right_bottom'
};

scan_points_v = [
     0.300,  0.450;   % 1 left_top
    -0.100,  0.450;   % 2 top_middle
    -0.510,  0.450;   % 3 right_top
     0.300,  0.025;   % 4 left_middle
    -0.090,  0.025;   % 5 center
    -0.510,  0.025;   % 6 right_middle
     0.300, -0.385;   % 7 left_bottom
    -0.085, -0.385;   % 8 bottom_middle
    -0.500, -0.385;   % 9 right_bottom
];

if any(abs(scan_points_v(:)) > V_LIMIT_V)
    error('Scan voltage exceeds safety range: %.3f V to %.3f V.', ...
        -V_LIMIT_V, V_LIMIT_V);
end

state = struct();
state.afg = [];
state.pf32 = [];
state.isGalvoOpen = false;
state.isSpadOpen = false;
state.isBusy = false;
state.outputDir = '';
state.metadataPath = '';
state.runFolderName = '';
state.latestPhaseLog = [];
state.latestPhaseName = '';

handles = buildUi();
refreshOutputFolderLabel();
refreshHardwareStatus();
logMessage('UI 已打开。先点击“初始化硬件”，再选择 OBJ/REF 并开始扫描。');

%% --- UI construction ---
    function handles = buildUi()
        handles.fig = figure( ...
            'Name', '3x3 Galvo + SPAD Grid Scan', ...
            'NumberTitle', 'off', ...
            'MenuBar', 'none', ...
            'ToolBar', 'none', ...
            'Resize', 'off', ...
            'Units', 'pixels', ...
            'Position', [120, 60, 1040, 820], ...
            'CloseRequestFcn', @onCloseFigure);

        handles.title = uicontrol(handles.fig, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [20, 782, 500, 24], ...
            'HorizontalAlignment', 'left', ...
            'FontSize', 13, 'FontWeight', 'bold', ...
            'String', '3x3 Galvo + SPAD Grid Scan');

        handles.status = uicontrol(handles.fig, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [555, 782, 460, 24], ...
            'HorizontalAlignment', 'right', ...
            'FontSize', 10, ...
            'String', '硬件状态：未初始化');

        handles.paramPanel = uipanel(handles.fig, ...
            'Units', 'pixels', ...
            'Title', '参数', ...
            'Position', [20, 365, 360, 400]);

        xLabel = 18;
        xEdit = 155;
        y0 = 342;
        dy = 32;
        editW = 170;
        addLabel(handles.paramPanel, xLabel, y0, 'AFG VISA');
        handles.afgVisaEdit = addEdit(handles.paramPanel, xEdit, y0, editW, DEFAULT_AFG_VISA_ADDRESS);

        addLabel(handles.paramPanel, xLabel, y0 - dy, '曝光时间 us');
        handles.expoEdit = addEdit(handles.paramPanel, xEdit, y0 - dy, editW, num2str(DEFAULT_EXPO_TIME_US));

        addLabel(handles.paramPanel, xLabel, y0 - 2*dy, '每个 hist 帧数');
        handles.framesEdit = addEdit(handles.paramPanel, xEdit, y0 - 2*dy, editW, num2str(DEFAULT_NUM_FRAMES));

        addLabel(handles.paramPanel, xLabel, y0 - 3*dy, '每点平均 hist 数');
        handles.repeatsEdit = addEdit(handles.paramPanel, xEdit, y0 - 3*dy, editW, num2str(DEFAULT_HIST_REPEATS));

        addLabel(handles.paramPanel, xLabel, y0 - 4*dy, '激光频率 Hz');
        handles.freqEdit = addEdit(handles.paramPanel, xEdit, y0 - 4*dy, editW, num2str(DEFAULT_LASER_REP_FREQ_HZ));

        addLabel(handles.paramPanel, xLabel, y0 - 5*dy, 'Galvo 稳定时间 s');
        handles.settleEdit = addEdit(handles.paramPanel, xEdit, y0 - 5*dy, editW, num2str(DEFAULT_GALVO_SETTLE_S));

        addLabel(handles.paramPanel, xLabel, y0 - 6*dy, 'Laser degree');
        handles.degreeEdit = addEdit(handles.paramPanel, xEdit, y0 - 6*dy, editW, num2str(DEFAULT_LASER_DEGREE));

        handles.displayHistCheck = uicontrol(handles.paramPanel, 'Style', 'checkbox', ...
            'Units', 'pixels', ...
            'Position', [18, 122, 150, 22], ...
            'Value', DEFAULT_DISPLAY_EACH_HIST, ...
            'String', '每点显示 hist');

        handles.overwriteCheck = uicontrol(handles.paramPanel, 'Style', 'checkbox', ...
            'Units', 'pixels', ...
            'Position', [175, 122, 150, 22], ...
            'Value', DEFAULT_ALLOW_OVERWRITE, ...
            'String', '允许覆盖');

        addLabel(handles.paramPanel, xLabel, 88, '实验备注');
        handles.noteEdit = uicontrol(handles.paramPanel, 'Style', 'edit', ...
            'Units', 'pixels', ...
            'Position', [18, 18, 320, 68], ...
            'HorizontalAlignment', 'left', ...
            'Max', 2, ...
            'Min', 0, ...
            'String', '');

        handles.modePanel = uibuttongroup(handles.fig, ...
            'Units', 'pixels', ...
            'Title', '采集模式', ...
            'Position', [400, 665, 210, 100]);

        handles.objRadio = uicontrol(handles.modePanel, 'Style', 'radiobutton', ...
            'Units', 'pixels', ...
            'Position', [18, 45, 160, 25], ...
            'String', 'OBJ 模式', ...
            'Tag', 'obj', ...
            'Value', 1);

        handles.refRadio = uicontrol(handles.modePanel, 'Style', 'radiobutton', ...
            'Units', 'pixels', ...
            'Position', [18, 15, 160, 25], ...
            'String', 'REF 模式', ...
            'Tag', 'ref');

        handles.controlPanel = uipanel(handles.fig, ...
            'Units', 'pixels', ...
            'Title', '控制', ...
            'Position', [630, 465, 390, 300]);

        handles.initButton = addButton(handles.controlPanel, [25, 235, 155, 36], ...
            '初始化硬件', @onInitHardware);
        handles.closeButton = addButton(handles.controlPanel, [205, 235, 155, 36], ...
            '关闭硬件', @onCloseHardware);

        handles.scanModeButton = addButton(handles.controlPanel, [25, 180, 335, 42], ...
            '扫描当前模式选中点', @onScanCurrentMode);

        handles.fullFlowButton = addButton(handles.controlPanel, [25, 125, 335, 42], ...
            '完整流程：OBJ → 提醒关激光 → REF', @onRunFullFlow);

        handles.showFolderButton = addButton(handles.controlPanel, [25, 72, 155, 34], ...
            '新建数据组', @onNewRunGroup);

        handles.visualizeButton = addButton(handles.controlPanel, [205, 72, 155, 34], ...
            '可视化最新一组', @onVisualizeLatest);

        handles.stopNote = uicontrol(handles.controlPanel, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [25, 45, 335, 18], ...
            'HorizontalAlignment', 'left', ...
            'String', '运行中如需停止，请在 MATLAB 里 Ctrl+C。');

        handles.folderText = uicontrol(handles.controlPanel, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [25, 5, 335, 36], ...
            'HorizontalAlignment', 'left', ...
            'String', '');

        handles.gridPanel = uipanel(handles.fig, ...
            'Units', 'pixels', ...
            'Title', '扫描点选择 / 顺序', ...
            'Position', [400, 465, 210, 180]);

        handles.pointChecks = gobjects(9, 1);
        handles.pointLabels = gobjects(9, 1);
        gridX = [20, 78, 136];
        gridY = [110, 62, 14];
        for idx = 1:9
            row = ceil(idx / 3);
            col = mod(idx - 1, 3) + 1;
            label = sprintf('%d\n%s', idx, scan_point_names{idx});
            handles.pointChecks(idx) = uicontrol(handles.gridPanel, 'Style', 'checkbox', ...
                'Units', 'pixels', ...
                'Position', [gridX(col), gridY(row), 52, 40], ...
                'BackgroundColor', [0.94, 0.94, 0.94], ...
                'HorizontalAlignment', 'center', ...
                'String', label, ...
                'Value', 1);
            handles.pointLabels(idx) = handles.pointChecks(idx);
        end

        handles.logBox = uicontrol(handles.fig, 'Style', 'listbox', ...
            'Units', 'pixels', ...
            'Position', [20, 20, 1000, 325], ...
            'FontName', 'Consolas', ...
            'FontSize', 9, ...
            'String', {});

        set(handles.modePanel, 'SelectedObject', handles.objRadio);
        set(handles.modePanel, 'SelectionChangedFcn', @onModeChanged);
    end

%% --- UI callbacks ---
    function onInitHardware(~, ~)
        if state.isBusy
            return;
        end

        setBusy(true);
        try
            cfg = readConfigFromUi();
            logMessage('正在初始化 Galvo 和 SPAD...');

            if state.isGalvoOpen
                closeAfgState();
            end
            if state.isSpadOpen
                closeSpadState();
            end

            state.afg = openAfg(cfg.afgVisaAddress);
            state.isGalvoOpen = true;

            state.pf32 = openSpad();
            state.isSpadOpen = true;
            configureSpad(state.pf32, cfg.expoTimeUs);

            refreshHardwareStatus();
            logMessage('硬件初始化完成。');
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
        logMessage('硬件已关闭。');
    end

    function onScanCurrentMode(~, ~)
        if state.isBusy
            return;
        end

        phaseName = getSelectedPhase();
        runOnePhaseFromUi(phaseName);
    end

    function onRunFullFlow(~, ~)
        if state.isBusy
            return;
        end

        set(handles.modePanel, 'SelectedObject', handles.objRadio);
        logMessage('已切换到 OBJ 模式。');
        ok = runOnePhaseFromUi('obj');
        if ~ok
            return;
        end

        uiwait(msgbox({ ...
            '9 个 OBJ 扫描点已经采集完成。', ...
            '请现在关闭激光器。', ...
            '关闭后点击确认，脚本会切换到 REF 模式并开始采集。' ...
            }, '关闭激光器', 'modal'));

        set(handles.modePanel, 'SelectedObject', handles.refRadio);
        logMessage('已切换到 REF 模式。');
        runOnePhaseFromUi('ref');
    end

    function onModeChanged(~, event)
        logMessage(sprintf('已切换到 %s 模式。', upper(get(event.NewValue, 'Tag'))));
    end

    function onNewRunGroup(~, ~)
        if state.isBusy
            return;
        end

        state.outputDir = '';
        state.metadataPath = '';
        state.runFolderName = '';
        state.latestPhaseLog = [];
        state.latestPhaseName = '';
        refreshOutputFolderLabel();
        logMessage('已准备新建数据组：下一次扫描会按当前参数生成新的唯一文件夹。');
    end

    function onVisualizeLatest(~, ~)
        if state.isBusy
            return;
        end

        try
            phaseLog = getLatestPhaseLogForVisualization();
            visualizeLatestGroup(phaseLog);
        catch ME
            logError(ME);
        end
    end

    function onCloseFigure(~, ~)
        if state.isBusy
            choice = questdlg('扫描/采集正在进行。确定要关闭 UI 吗？', ...
                '关闭确认', '关闭', '取消', '取消');
            if ~strcmp(choice, '关闭')
                return;
            end
        end

        closeHardwareQuietly();
        delete(handles.fig);
    end

%% --- Scan workflow ---
    function ok = runOnePhaseFromUi(phaseName)
        ok = false;
        setBusy(true);

        try
            ensureHardwareReady();
            cfg = readConfigFromUi();
            configureSpad(state.pf32, cfg.expoTimeUs);

            [outputDir, metadataPath] = prepareOutputFolder(cfg);
            state.outputDir = outputDir;
            state.metadataPath = metadataPath;
            refreshOutputFolderLabel();

            ensurePhaseOutputsAvailable(phaseName, cfg, outputDir);

            phaseLog = acquireGridPhase( ...
                phaseName, state.pf32, state.afg, ...
                scan_point_names, scan_point_ids, scan_points_v, cfg, ...
                outputDir);

            phaseLogName = sprintf('%s_scan_log.mat', phaseName);
            phaseLogPath = fullfile(outputDir, phaseLogName);

            if strcmp(phaseName, 'obj')
                obj_log = phaseLog; %#ok<NASGU>
                save(phaseLogPath, 'obj_log');
            else
                ref_log = phaseLog; %#ok<NASGU>
                save(phaseLogPath, 'ref_log');
            end

            trySaveCombinedScanLog(outputDir, metadataPath);
            state.latestPhaseLog = phaseLog;
            state.latestPhaseName = phaseName;

            logMessage(sprintf('%s 模式扫描完成。', upper(phaseName)));
            ok = true;
        catch ME
            logError(ME);
        end

        resetPointHighlights();
        setBusy(false);
    end

    function phaseLog = acquireGridPhase(phaseName, pf32, afg, ...
            pointNames, pointIds, scanPointsV, cfg, outputDir)

        selectedPointIndices = cfg.selectedPointIndices;
        numSelectedPoints = numel(selectedPointIndices);
        phaseLog = repmat(blankScanInfo(), numSelectedPoints, 1);

        for selectedIdx = 1:numSelectedPoints
            pointIdx = selectedPointIndices(selectedIdx);
            highlightPoint(pointIdx, [1.00, 0.93, 0.55]);

            targetVoltageX = scanPointsV(pointIdx, 1);
            targetVoltageY = scanPointsV(pointIdx, 2);

            writeGalvoVoltage(afg, targetVoltageX, targetVoltageY);

            if cfg.galvoSettleS > 0
                pause(cfg.galvoSettleS);
            end

            logMessage(sprintf('[%s] 点 %d/%d：实际点 %d (%s), X=%.3f V, Y=%.3f V，开始采集 %d 个 hist 做平均。', ...
                upper(phaseName), selectedIdx, numSelectedPoints, pointIdx, pointNames{pointIdx}, ...
                targetVoltageX, targetVoltageY, cfg.histRepeats));

            tic;
            [hist, timeAxis_ns, binEdges_ns, repeatElapsedS] = acquireAverageHist( ...
                pf32, cfg.numFrames, cfg.laserRepFreqHz, cfg.histRepeats, phaseName, pointIdx);
            elapsedS = toc;

            scanInfo = blankScanInfo();
            scanInfo.phase = phaseName;
            scanInfo.point_index = pointIdx;
            scanInfo.point_name = pointNames{pointIdx};
            scanInfo.point_id = pointIds{pointIdx};
            scanInfo.target_voltage_x_v = targetVoltageX;
            scanInfo.target_voltage_y_v = targetVoltageY;
            scanInfo.expo_time_us = cfg.expoTimeUs;
            scanInfo.num_frames = cfg.numFrames;
            scanInfo.hist_repeats = cfg.histRepeats;
            scanInfo.laser_rep_freq_hz = cfg.laserRepFreqHz;
            scanInfo.laser_degree = cfg.laserDegree;
            scanInfo.elapsed_s = elapsedS;
            scanInfo.repeat_elapsed_s = repeatElapsedS;
            scanInfo.acquired_at = timestampText();

            fileName = sprintf('hist_%dus_%g_avg%d_point%02d_%s_%s.mat', ...
                cfg.expoTimeUs, cfg.numFrames, cfg.histRepeats, ...
                pointIdx, pointIds{pointIdx}, phaseName);
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

            clear hist;
            phaseLog(selectedIdx, 1) = scanInfo;
            highlightPoint(pointIdx, [0.70, 0.90, 0.70]);

            logMessage(sprintf('[%s] 已保存：%s，总耗时 %.2f s。', ...
                upper(phaseName), fileName, elapsedS));
        end
    end

    function ensurePhaseOutputsAvailable(phaseName, cfg, outputDir)
        if cfg.allowOverwrite
            return;
        end

        phaseLogPath = fullfile(outputDir, sprintf('%s_scan_log.mat', phaseName));
        if exist(phaseLogPath, 'file')
            error('Phase log already exists: %s', phaseLogPath);
        end

        for pointIdx = reshape(cfg.selectedPointIndices, 1, [])
            fileName = sprintf('hist_%dus_%g_avg%d_point%02d_%s_%s.mat', ...
                cfg.expoTimeUs, cfg.numFrames, cfg.histRepeats, ...
                pointIdx, scan_point_ids{pointIdx}, phaseName);
            filePath = fullfile(outputDir, fileName);
            if exist(filePath, 'file')
                error('Output file already exists: %s', filePath);
            end
        end
    end

    function [histAvg, timeAxis_ns, binEdges_ns, repeatElapsedS] = acquireAverageHist( ...
            pf32, numFrames, laserRepFreqHz, histRepeats, phaseName, pointIdx)

        histSum = [];
        timeAxis_ns = [];
        binEdges_ns = [];
        repeatElapsedS = zeros(histRepeats, 1);

        for repIdx = 1:histRepeats
            logMessage(sprintf('[%s] 点 %d: hist %d/%d，采集 %.0f 帧...', ...
                upper(phaseName), pointIdx, repIdx, histRepeats, numFrames));

            tic;
            series = pf_getMultipleFrames(pf32, numFrames);
            repeatElapsedS(repIdx) = toc;

            [histOne, timeAxis_ns, binEdges_ns] = series2hist(series, laserRepFreqHz);
            clear series;

            if isempty(histSum)
                histSum = double(histOne);
            else
                histSum = histSum + double(histOne);
            end

            clear histOne;
        end

        histAvg = histSum ./ histRepeats;
    end

%% --- Config / metadata ---
    function cfg = readConfigFromUi()
        cfg = struct();
        cfg.afgVisaAddress = strtrim(get(handles.afgVisaEdit, 'String'));
        cfg.expoTimeUs = readPositiveNumber(handles.expoEdit, '曝光时间 us');
        cfg.numFrames = round(readPositiveNumber(handles.framesEdit, '每个 hist 帧数'));
        cfg.histRepeats = round(readPositiveNumber(handles.repeatsEdit, '每点平均 hist 数'));
        cfg.laserRepFreqHz = readPositiveNumber(handles.freqEdit, '激光频率 Hz');
        cfg.galvoSettleS = readNonnegativeNumber(handles.settleEdit, 'Galvo 稳定时间 s');
        cfg.laserDegree = readFiniteNumber(handles.degreeEdit, 'Laser degree');
        cfg.displayEachHist = logical(get(handles.displayHistCheck, 'Value'));
        cfg.allowOverwrite = logical(get(handles.overwriteCheck, 'Value'));
        cfg.experimentNote = readMultilineText(handles.noteEdit);
        cfg.selectedPointIndices = readSelectedPointIndices();

        if isempty(cfg.afgVisaAddress)
            error('AFG VISA 地址不能为空。');
        end
        if cfg.histRepeats < 1
            error('每点平均 hist 数必须 >= 1。');
        end
        if cfg.numFrames < 1
            error('每个 hist 帧数必须 >= 1。');
        end
        if isempty(cfg.selectedPointIndices)
            error('至少需要选择一个扫描点。');
        end
    end

    function [outputDir, metadataPath] = prepareOutputFolder(cfg)
        dataRoot = fullfile(PROJECT_ROOT, 'data');
        if ~exist(dataRoot, 'dir')
            mkdir(dataRoot);
        end

        if isempty(state.outputDir)
            [state.outputDir, state.runFolderName] = reserveOutputFolder(dataRoot, cfg);
            state.metadataPath = fullfile(state.outputDir, 'scan_metadata.mat');
        end

        outputDir = state.outputDir;
        metadataPath = state.metadataPath;

        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end

        metadata = struct();
        metadata.created_at = timestampText();
        metadata.run_folder_name = state.runFolderName;
        metadata.project_root = char(PROJECT_ROOT);
        metadata.output_dir = char(outputDir);
        metadata.afg_visa_address = cfg.afgVisaAddress;
        metadata.afg_output_load = 'High-Z / MAX';
        metadata.park_voltage_v = [PARK_X_V, PARK_Y_V];
        metadata.laser_degree = cfg.laserDegree;
        metadata.expo_time_us = cfg.expoTimeUs;
        metadata.num_frames = cfg.numFrames;
        metadata.hist_repeats = cfg.histRepeats;
        metadata.laser_rep_freq_hz = cfg.laserRepFreqHz;
        metadata.galvo_settle_s = cfg.galvoSettleS;
        metadata.allow_overwrite = cfg.allowOverwrite;
        metadata.experiment_note = cfg.experimentNote;
        metadata.note_file = char(fullfile(outputDir, 'README_note.txt'));
        metadata.selected_point_indices = cfg.selectedPointIndices;
        metadata.scan_point_names = scan_point_names;
        metadata.scan_point_ids = scan_point_ids;
        metadata.scan_points_v = scan_points_v;

        if exist(metadataPath, 'file') && ~cfg.allowOverwrite
            existingVars = whos('-file', metadataPath);
            hasMetadata = any(strcmp({existingVars.name}, 'metadata'));
            hasScanLog = any(strcmp({existingVars.name}, 'scan_log'));
            if ~hasMetadata
                error('Metadata file exists but has unexpected content: %s', metadataPath);
            end
            if hasScanLog
                error(['Output folder already contains a finished scan: %s\n' ...
                    'Enable "允许覆盖" or rename/move the existing folder.'], outputDir);
            end
        end

        save(metadataPath, 'metadata');
        writeExperimentNoteFile(outputDir, metadata, cfg.experimentNote);
    end

    function trySaveCombinedScanLog(outputDir, metadataPath)
        objLogPath = fullfile(outputDir, 'obj_scan_log.mat');
        refLogPath = fullfile(outputDir, 'ref_scan_log.mat');

        if exist(objLogPath, 'file') && exist(refLogPath, 'file')
            objData = load(objLogPath, 'obj_log');
            refData = load(refLogPath, 'ref_log');
            scan_log = [objData.obj_log; refData.ref_log]; %#ok<NASGU>

            metaData = load(metadataPath, 'metadata');
            metadata = metaData.metadata;
            metadata.finished_at = timestampText();

            save(metadataPath, 'metadata', 'scan_log');
            logMessage('OBJ 和 REF 日志都已存在，已更新 scan_metadata.mat。');
        end
    end

    function [outputDir, runFolderName] = reserveOutputFolder(dataRoot, cfg)
        baseFolderName = makeRunFolderName(cfg);
        runFolderName = baseFolderName;
        outputDir = fullfile(dataRoot, runFolderName);

        suffix = 1;
        while exist(outputDir, 'dir')
            suffix = suffix + 1;
            runFolderName = sprintf('%s_r%02d', baseFolderName, suffix);
            outputDir = fullfile(dataRoot, runFolderName);
        end
    end

    function runFolderName = makeRunFolderName(cfg)
        stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
        runFolderName = sprintf('3x3_grid_scan_%s_deg_%s_exp_%sus_frames_%d_avg_%d', ...
            stamp, numberToken(cfg.laserDegree), numberToken(cfg.expoTimeUs), ...
            cfg.numFrames, cfg.histRepeats);
    end

    function writeExperimentNoteFile(outputDir, metadata, experimentNote)
        notePath = fullfile(outputDir, 'README_note.txt');
        fid = fopen(notePath, 'w', 'n', 'UTF-8');
        if fid < 0
            error('Cannot write experiment note file: %s', notePath);
        end

        cleaner = onCleanup(@() fclose(fid));
        fprintf(fid, 'Experiment note\n');
        fprintf(fid, 'Created at: %s\n', metadata.created_at);
        fprintf(fid, 'Run folder: %s\n', metadata.run_folder_name);
        fprintf(fid, 'Output dir: %s\n', metadata.output_dir);
        fprintf(fid, '\n');

        if isempty(strtrim(experimentNote))
            fprintf(fid, '(No note entered.)\n');
        else
            fprintf(fid, '%s\n', experimentNote);
        end

        clear cleaner;
    end

    function token = numberToken(value)
        if value < 0
            prefix = 'neg';
        else
            prefix = '';
        end

        token = sprintf('%.6g', abs(value));
        token = strrep(token, '.', 'p');
        token = strrep(token, '+', '');
        token = [prefix token];
    end

%% --- UI helpers ---
    function phaseName = getSelectedPhase()
        selectedObj = get(handles.modePanel, 'SelectedObject');
        phaseName = get(selectedObj, 'Tag');
    end

    function selectedPointIndices = readSelectedPointIndices()
        selectedPointIndices = [];
        for pointIdx = 1:numel(handles.pointChecks)
            if logical(get(handles.pointChecks(pointIdx), 'Value'))
                selectedPointIndices(end + 1) = pointIdx; %#ok<AGROW>
            end
        end
    end

    function phaseLog = getLatestPhaseLogForVisualization()
        if ~isempty(state.latestPhaseLog)
            phaseLog = state.latestPhaseLog;
            return;
        end

        refreshOutputFolderLabel();
        currentPhase = getSelectedPhase();
        candidatePhases = {currentPhase, 'ref', 'obj'};

        for phaseIdx = 1:numel(candidatePhases)
            phaseName = candidatePhases{phaseIdx};
            logPath = fullfile(state.outputDir, sprintf('%s_scan_log.mat', phaseName));
            if exist(logPath, 'file')
                logData = load(logPath);
                logVarName = sprintf('%s_log', phaseName);
                if isfield(logData, logVarName)
                    phaseLog = logData.(logVarName);
                    state.latestPhaseLog = phaseLog;
                    state.latestPhaseName = phaseName;
                    return;
                end
            end
        end

        error('还没有可视化的数据。请先完成一次 OBJ 或 REF 扫描。');
    end

    function visualizeLatestGroup(phaseLog)
        if isempty(phaseLog)
            error('最新扫描记录为空，无法可视化。');
        end

        figName = sprintf('Latest %s hist group: pixel (16,16)', upper(state.latestPhaseName));
        fig = figure( ...
            'Name', figName, ...
            'NumberTitle', 'off', ...
            'Color', 'w', ...
            'Position', [160, 80, 1180, 820]);

        for pointIdx = 1:9
            ax = subplot(3, 3, pointIdx, 'Parent', fig);
            logIdx = find([phaseLog.point_index] == pointIdx, 1, 'last');

            if isempty(logIdx)
                axis(ax, 'off');
                text(ax, 0.5, 0.5, sprintf('%d %s\n未扫描', pointIdx, scan_point_names{pointIdx}), ...
                    'HorizontalAlignment', 'center', ...
                    'FontSize', 11);
                continue;
            end

            filePath = phaseLog(logIdx).file;
            if ~exist(filePath, 'file')
                axis(ax, 'off');
                text(ax, 0.5, 0.5, sprintf('%d %s\n文件不存在', pointIdx, scan_point_names{pointIdx}), ...
                    'HorizontalAlignment', 'center', ...
                    'FontSize', 11);
                continue;
            end

            histData = load(filePath, 'hist', 'timeAxis_ns');
            histCube = double(histData.hist);
            histCube(~isfinite(histCube)) = 0;

            yPix = min(16, size(histCube, 1));
            xPix = min(16, size(histCube, 2));
            curve = squeeze(histCube(yPix, xPix, :));

            if isfield(histData, 'timeAxis_ns') && numel(histData.timeAxis_ns) == numel(curve)
                xAxis = histData.timeAxis_ns(:);
                xLabelText = 'Time (ns)';
            else
                xAxis = (1:numel(curve)).';
                xLabelText = 'Bin';
            end

            hLine = plot(ax, xAxis, curve, 'LineWidth', 1.2);
            grid(ax, 'on');
            title(ax, sprintf('%d %s', pointIdx, scan_point_names{pointIdx}), 'Interpreter', 'none');
            xlabel(ax, xLabelText);
            ylabel(ax, 'Counts');

            callback = @(~, ~) openFullHistFromFile(filePath);
            set(ax, 'ButtonDownFcn', callback);
            set(hLine, 'ButtonDownFcn', callback);
        end

        sgtitle(fig, sprintf('Latest %s scan: pixel (16,16). Click a subplot to open my\\_display\\_hist.', ...
            upper(state.latestPhaseName)), 'Interpreter', 'none');

        logMessage('已打开最新一组 hist 的 3x3 可视化窗口。点击任意已扫描子图可打开完整 my_display_hist。');
    end

    function openFullHistFromFile(filePath)
        try
            histData = load(filePath, 'hist');
            my_display_hist(histData.hist);
        catch ME
            logError(ME);
        end
    end

    function refreshOutputFolderLabel()
        if isempty(state.outputDir)
            set(handles.folderText, 'String', sprintf('保存目录:\n首次扫描时自动生成；点“新建数据组”可重置'));
        else
            set(handles.folderText, 'String', sprintf('保存目录:\n%s', state.outputDir));
        end
    end

    function refreshHardwareStatus()
        statusText = sprintf('硬件状态：Galvo %s / SPAD %s', ...
            onOffText(state.isGalvoOpen), onOffText(state.isSpadOpen));
        set(handles.status, 'String', statusText);
    end

    function setBusy(isBusy)
        state.isBusy = isBusy;
        controls = [
            handles.initButton
            handles.closeButton
            handles.scanModeButton
            handles.fullFlowButton
            handles.showFolderButton
            handles.visualizeButton
        ];

        if isBusy
            set(controls, 'Enable', 'off');
            set(handles.status, 'String', '硬件状态：运行中...');
        else
            set(controls, 'Enable', 'on');
            refreshHardwareStatus();
        end
        drawnow;
    end

    function highlightPoint(idx, color)
        resetPointHighlights();
        set(handles.pointLabels(idx), 'BackgroundColor', color);
        drawnow;
    end

    function resetPointHighlights()
        for pointIdx = 1:numel(handles.pointLabels)
            if ishandle(handles.pointLabels(pointIdx))
                set(handles.pointLabels(pointIdx), 'BackgroundColor', [0.94, 0.94, 0.94]);
            end
        end
        drawnow;
    end

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
        logMessage(sprintf('错误：%s', ME.message));
        for stackIdx = 1:numel(ME.stack)
            logMessage(sprintf('  at %s line %d', ME.stack(stackIdx).name, ME.stack(stackIdx).line));
        end
    end

    function ensureHardwareReady()
        if ~state.isGalvoOpen || isempty(state.afg)
            error('AFG/Galvo 尚未初始化，请先点击“初始化硬件”。');
        end
        if ~state.isSpadOpen || isempty(state.pf32)
            error('SPAD 尚未初始化，请先点击“初始化硬件”。');
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
end

%% --- Hardware helpers ---
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

function validateAfgVoltage(voltageV, axisName)
    maxAbsVoltageV = 2;
    if ~isfinite(voltageV) || abs(voltageV) > maxAbsVoltageV
        error('Galvo %s voltage %.6g V exceeds safety range +/- %.3f V.', ...
            axisName, voltageV, maxAbsVoltageV);
    end
end

function setAfgDcVoltage(afg, chan, voltageV)
    writeline(afg, sprintf('SOUR%d:VOLT:OFFS %.9gV', chan, voltageV));
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

%% --- Data / validation helpers ---
function scanInfo = blankScanInfo()
    scanInfo = struct( ...
        'phase', '', ...
        'point_index', [], ...
        'point_name', '', ...
        'point_id', '', ...
        'target_voltage_x_v', [], ...
        'target_voltage_y_v', [], ...
        'expo_time_us', [], ...
        'num_frames', [], ...
        'hist_repeats', [], ...
        'laser_rep_freq_hz', [], ...
        'laser_degree', [], ...
        'elapsed_s', [], ...
        'repeat_elapsed_s', [], ...
        'acquired_at', '', ...
        'file', '');
end

function text = timestampText()
    text = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
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
        error('%s 必须是正数。', label);
    end
end

function value = readNonnegativeNumber(editHandle, label)
    value = str2double(strtrim(get(editHandle, 'String')));
    if ~isfinite(value) || value < 0
        error('%s 必须是非负数。', label);
    end
end

function value = readFiniteNumber(editHandle, label)
    value = str2double(strtrim(get(editHandle, 'String')));
    if ~isfinite(value)
        error('%s 必须是有效数字。', label);
    end
end

function text = onOffText(isOn)
    if isOn
        text = '已连接';
    else
        text = '未连接';
    end
end

function addLabel(parent, x, y, labelText)
    uicontrol(parent, 'Style', 'text', ...
        'Units', 'pixels', ...
        'Position', [x, y, 125, 22], ...
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
