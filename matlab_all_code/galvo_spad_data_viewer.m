function galvo_spad_data_viewer()
%GALVO_SPAD_DATA_VIEWER Visualize saved 3x3 galvo + SPAD histogram scans.
%   Select a saved data folder, choose OBJ / REF / abs(OBJ-REF), choose
%   rows and columns to sum, then view the 9-point overview. Click any
%   subplot to open the full 32x32 my_display_hist view for that point.

PROJECT_ROOT = fileparts(mfilename('fullpath'));
addpath(genpath(PROJECT_ROOT));

DATA_ROOT = fullfile(PROJECT_ROOT, 'data');

pointNames = {
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

state = struct();
state.folder = DATA_ROOT;
state.files = blankFileMap();
state.latestMode = 'obj';
state.latestRows = 16;
state.latestCols = 16;

handles = buildUi();
logMessage('请选择一个已采集数据文件夹，然后点击“加载文件夹”。');

%% --- UI construction ---
    function handles = buildUi()
        handles.fig = figure( ...
            'Name', 'Galvo SPAD Data Viewer', ...
            'NumberTitle', 'off', ...
            'MenuBar', 'none', ...
            'ToolBar', 'none', ...
            'Resize', 'off', ...
            'Units', 'pixels', ...
            'Position', [80, 60, 1320, 820]);

        handles.title = uicontrol(handles.fig, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [20, 782, 420, 24], ...
            'HorizontalAlignment', 'left', ...
            'FontSize', 13, ...
            'FontWeight', 'bold', ...
            'String', 'Galvo SPAD Data Viewer');

        handles.folderEdit = uicontrol(handles.fig, 'Style', 'edit', ...
            'Units', 'pixels', ...
            'Position', [20, 742, 760, 28], ...
            'HorizontalAlignment', 'left', ...
            'String', state.folder);

        handles.browseButton = uicontrol(handles.fig, 'Style', 'pushbutton', ...
            'Units', 'pixels', ...
            'Position', [790, 742, 95, 28], ...
            'String', '选择文件夹', ...
            'Callback', @onBrowseFolder);

        handles.loadButton = uicontrol(handles.fig, 'Style', 'pushbutton', ...
            'Units', 'pixels', ...
            'Position', [895, 742, 95, 28], ...
            'String', '加载文件夹', ...
            'Callback', @onLoadFolder);

        handles.plotButton = uicontrol(handles.fig, 'Style', 'pushbutton', ...
            'Units', 'pixels', ...
            'Position', [1000, 742, 135, 28], ...
            'String', '生成9点总览', ...
            'Callback', @onPlotOverview);

        handles.openFolderButton = uicontrol(handles.fig, 'Style', 'pushbutton', ...
            'Units', 'pixels', ...
            'Position', [1145, 742, 135, 28], ...
            'String', '打开数据文件夹', ...
            'Callback', @onOpenFolder);

        handles.controlPanel = uipanel(handles.fig, ...
            'Units', 'pixels', ...
            'Title', '显示设置', ...
            'Position', [20, 625, 1260, 100]);

        uicontrol(handles.controlPanel, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [18, 42, 80, 22], ...
            'HorizontalAlignment', 'left', ...
            'String', '显示模式');

        handles.modePopup = uicontrol(handles.controlPanel, 'Style', 'popupmenu', ...
            'Units', 'pixels', ...
            'Position', [95, 42, 155, 26], ...
            'String', {'OBJ'}, ...
            'Callback', @onPlotOverview);

        uicontrol(handles.controlPanel, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [280, 42, 110, 22], ...
            'HorizontalAlignment', 'left', ...
            'String', '相加行索引 Y');

        handles.rowEdit = uicontrol(handles.controlPanel, 'Style', 'edit', ...
            'Units', 'pixels', ...
            'Position', [390, 42, 150, 26], ...
            'HorizontalAlignment', 'left', ...
            'String', '16');

        uicontrol(handles.controlPanel, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [565, 42, 110, 22], ...
            'HorizontalAlignment', 'left', ...
            'String', '相加列索引 X');

        handles.colEdit = uicontrol(handles.controlPanel, 'Style', 'edit', ...
            'Units', 'pixels', ...
            'Position', [675, 42, 150, 26], ...
            'HorizontalAlignment', 'left', ...
            'String', '16');

        handles.autoYCheck = uicontrol(handles.controlPanel, 'Style', 'checkbox', ...
            'Units', 'pixels', ...
            'Position', [850, 42, 110, 24], ...
            'String', '统一Y轴', ...
            'Value', 0);

        handles.logScaleCheck = uicontrol(handles.controlPanel, 'Style', 'checkbox', ...
            'Units', 'pixels', ...
            'Position', [970, 42, 110, 24], ...
            'String', 'Y轴log', ...
            'Value', 0);

        handles.saveCalButton = uicontrol(handles.controlPanel, 'Style', 'pushbutton', ...
            'Units', 'pixels', ...
            'Position', [1100, 42, 135, 26], ...
            'String', '保存 CAL', ...
            'Enable', 'off', ...
            'Callback', @onSaveCalFiles);

        handles.intensityOverviewButton = uicontrol(handles.controlPanel, 'Style', 'pushbutton', ...
            'Units', 'pixels', ...
            'Position', [1100, 10, 135, 26], ...
            'String', '9点强度图', ...
            'Enable', 'off', ...
            'Callback', @onShowIntensityOverview);

        handles.statusText = uicontrol(handles.controlPanel, 'Style', 'text', ...
            'Units', 'pixels', ...
            'Position', [18, 10, 1065, 22], ...
            'HorizontalAlignment', 'left', ...
            'String', '未加载数据');

        handles.axesPanel = uipanel(handles.fig, ...
            'Units', 'pixels', ...
            'Title', '9点直方图曲线总览', ...
            'Position', [20, 95, 1260, 510]);

        handles.axes = gobjects(9, 1);
        lefts = [45, 450, 855];
        bottoms = [345, 185, 25];
        for pointIdx = 1:9
            row = ceil(pointIdx / 3);
            col = mod(pointIdx - 1, 3) + 1;
            handles.axes(pointIdx) = axes( ...
                'Parent', handles.axesPanel, ...
                'Units', 'pixels', ...
                'Position', [lefts(col), bottoms(row), 340, 120], ...
                'Box', 'on');
            title(handles.axes(pointIdx), sprintf('%d %s', pointIdx, pointNames{pointIdx}));
        end

        handles.logBox = uicontrol(handles.fig, 'Style', 'listbox', ...
            'Units', 'pixels', ...
            'Position', [20, 15, 1260, 65], ...
            'FontName', 'Consolas', ...
            'FontSize', 9, ...
            'String', {});
    end

%% --- UI callbacks ---
    function onBrowseFolder(~, ~)
        startDir = strtrim(get(handles.folderEdit, 'String'));
        if ~exist(startDir, 'dir')
            startDir = DATA_ROOT;
        end

        selectedFolder = uigetdir(startDir, '选择已采集数据文件夹');
        if isequal(selectedFolder, 0)
            return;
        end

        set(handles.folderEdit, 'String', selectedFolder);
        onLoadFolder();
    end

    function onLoadFolder(~, ~)
        folder = strtrim(get(handles.folderEdit, 'String'));
        if ~exist(folder, 'dir')
            logMessage(sprintf('错误：文件夹不存在：%s', folder));
            return;
        end

        state.folder = folder;
        state.files = detectHistFiles(folder);
        updateModePopup();
        updateStatus();
        logMessage(sprintf('已加载文件夹：%s', folder));
        onPlotOverview();
    end

    function onOpenFolder(~, ~)
        folder = strtrim(get(handles.folderEdit, 'String'));
        if exist(folder, 'dir')
            winopen(folder);
        else
            logMessage(sprintf('错误：文件夹不存在：%s', folder));
        end
    end

    function onSaveCalFiles(~, ~)
        try
            if ~hasAnyPairData(state.files)
                logMessage('没有可保存的 CAL 数据：需要同一扫描点同时有 OBJ 和 REF。');
                return;
            end

            pairPoints = find(arrayfun(@(s) ~isempty(s.obj) && ~isempty(s.ref), state.files));
            existingCalFiles = {};
            for pairIdx = 1:numel(pairPoints)
                pointIdx = pairPoints(pairIdx);
                calFileName = makeCalFileName(state.files(pointIdx).obj.name, pointIdx);
                calFilePath = fullfile(state.folder, calFileName);
                if exist(calFilePath, 'file')
                    existingCalFiles{end + 1} = calFileName; %#ok<AGROW>
                end
            end

            overwriteExisting = true;
            if ~isempty(existingCalFiles)
                choice = questdlg( ...
                    sprintf('已有 %d 个 CAL 文件。是否覆盖？', numel(existingCalFiles)), ...
                    '保存 CAL', ...
                    '覆盖', '跳过已有', '取消', '覆盖');
                if isempty(choice) || strcmp(choice, '取消')
                    logMessage('已取消保存 CAL。');
                    return;
                end
                overwriteExisting = strcmp(choice, '覆盖');
            end

            savedCount = 0;
            skippedCount = 0;
            for pairIdx = 1:numel(pairPoints)
                pointIdx = pairPoints(pairIdx);
                calFileName = makeCalFileName(state.files(pointIdx).obj.name, pointIdx);
                calFilePath = fullfile(state.folder, calFileName);

                if exist(calFilePath, 'file') && ~overwriteExisting
                    skippedCount = skippedCount + 1;
                    logMessage(sprintf('跳过已有 CAL：%s', calFileName));
                    continue;
                end

                saveCalPointFile(pointIdx, calFilePath);
                savedCount = savedCount + 1;
                logMessage(sprintf('已保存 CAL 点 %d (%s)：%s', ...
                    pointIdx, pointNames{pointIdx}, calFileName));
            end

            logMessage(sprintf('CAL 保存完成：保存 %d 个，跳过 %d 个。', savedCount, skippedCount));
        catch ME
            logError(ME);
        end
    end

    function onShowIntensityOverview(~, ~)
        try
            if ~hasAnyData(state.files)
                logMessage('没有可显示的强度图数据，请先选择并加载数据文件夹。');
                return;
            end

            modeName = getSelectedMode();
            intensityImages = cell(9, 1);
            fileLabels = cell(9, 1);
            globalMax = 0;

            for pointIdx = 1:9
                [histCube, ~, fileLabel] = loadPointHist(pointIdx, modeName);
                if isempty(histCube)
                    continue;
                end

                intensityImage = sum(histCube, 3);
                intensityImage(~isfinite(intensityImage)) = 0;
                intensityImages{pointIdx} = intensityImage;
                fileLabels{pointIdx} = fileLabel;
                globalMax = max(globalMax, max(intensityImage(:)));
            end

            fig = figure( ...
                'Name', sprintf('3x3 intensity images | %s', upper(modeName)), ...
                'NumberTitle', 'off', ...
                'Color', 'w');
            set(fig, 'Position', [180, 80, 940, 820]);

            layout = tiledlayout(fig, 3, 3, ...
                'TileSpacing', 'compact', ...
                'Padding', 'compact');
            title(layout, sprintf('3x3 intensity images | %s', upper(modeName)), ...
                'Interpreter', 'none');

            for pointIdx = 1:9
                ax = nexttile(layout, pointIdx);
                intensityImage = intensityImages{pointIdx};

                if isempty(intensityImage)
                    axis(ax, 'off');
                    text(ax, 0.5, 0.5, sprintf('%d %s\n无数据', pointIdx, pointNames{pointIdx}), ...
                        'Units', 'normalized', ...
                        'HorizontalAlignment', 'center', ...
                        'FontSize', 10);
                    continue;
                end

                hImg = imagesc(ax, intensityImage);
                axis(ax, 'image');
                colormap(ax, 'jet');
                colorbar(ax);

                if globalMax > 0
                    set(ax, 'CLim', [0, globalMax]);
                end

                title(ax, sprintf('%d %s', pointIdx, pointNames{pointIdx}), ...
                    'Interpreter', 'none');
                xlabel(ax, 'X pixel');
                ylabel(ax, 'Y pixel');
                set(ax, ...
                    'XLim', [0.5, size(intensityImage, 2) + 0.5], ...
                    'YLim', [0.5, size(intensityImage, 1) + 0.5], ...
                    'XTick', unique([1, round(size(intensityImage, 2) / 2), size(intensityImage, 2)]), ...
                    'YTick', unique([1, round(size(intensityImage, 1) / 2), size(intensityImage, 1)]));

                callback = @(~, ~) openFullPointView(pointIdx, modeName);
                set(ax, 'ButtonDownFcn', callback);
                set(hImg, 'ButtonDownFcn', callback);
                set(ax, 'UserData', struct( ...
                    'pointIdx', pointIdx, ...
                    'modeName', modeName, ...
                    'fileLabel', fileLabels{pointIdx}));
            end

            logMessage(sprintf('已打开 %s 模式 3x3 强度图总览。', upper(modeName)));
        catch ME
            logError(ME);
        end
    end

    function onPlotOverview(~, ~)
        try
            if ~hasAnyData(state.files)
                logMessage('没有可画的数据，请先选择并加载数据文件夹。');
                return;
            end

            modeName = getSelectedMode();
            [ny, nx] = getFirstAvailableHistSize(modeName);
            rowIdx = parseIndexList(get(handles.rowEdit, 'String'), ny, '行索引');
            colIdx = parseIndexList(get(handles.colEdit, 'String'), nx, '列索引');

            state.latestMode = modeName;
            state.latestRows = rowIdx;
            state.latestCols = colIdx;

            yMax = 0;
            plottedCurves = cell(9, 1);
            plottedAxes = cell(9, 1);

            for pointIdx = 1:9
                ax = handles.axes(pointIdx);
                cla(ax);

                [histCube, timeAxis_ns, fileLabel] = loadPointHist(pointIdx, modeName);
                if isempty(histCube)
                    axis(ax, 'off');
                    text(ax, 0.5, 0.5, sprintf('%d %s\n无数据', pointIdx, pointNames{pointIdx}), ...
                        'Parent', ax, ...
                        'Units', 'normalized', ...
                        'HorizontalAlignment', 'center', ...
                        'FontSize', 10);
                    continue;
                end

                axis(ax, 'on');
                curve = sumSelectedCurve(histCube, rowIdx, colIdx);
                [xAxis, xLabelText] = makeTimeAxis(curve, timeAxis_ns);

                plottedCurves{pointIdx} = curve;
                plottedAxes{pointIdx} = xAxis;
                yMax = max(yMax, max(curve));

                hLine = plot(ax, xAxis, curve, 'LineWidth', 1.25);
                grid(ax, 'on');
                title(ax, sprintf('%d %s', pointIdx, pointNames{pointIdx}), 'Interpreter', 'none');
                xlabel(ax, xLabelText);
                ylabel(ax, 'Counts');

                if get(handles.logScaleCheck, 'Value')
                    set(ax, 'YScale', 'log');
                    minPositive = min(curve(curve > 0));
                    if isempty(minPositive)
                        minPositive = 1;
                    end
                    ylim(ax, [minPositive, max(minPositive * 10, max(curve) * 1.2)]);
                else
                    set(ax, 'YScale', 'linear');
                end

                set(ax, 'UserData', struct( ...
                    'pointIdx', pointIdx, ...
                    'modeName', modeName, ...
                    'fileLabel', fileLabel));

                callback = @(~, ~) openFullPointView(pointIdx, modeName);
                set(ax, 'ButtonDownFcn', callback);
                set(hLine, 'ButtonDownFcn', callback);
            end

            if get(handles.autoYCheck, 'Value') && yMax > 0 && ~get(handles.logScaleCheck, 'Value')
                for pointIdx = 1:9
                    if ~isempty(plottedCurves{pointIdx})
                        ylim(handles.axes(pointIdx), [0, yMax * 1.1]);
                    end
                end
            end

            set(handles.statusText, 'String', sprintf( ...
                '模式: %s | 行Y: %s | 列X: %s | 点击任意子图打开完整32x32视图', ...
                upper(modeName), mat2str(rowIdx), mat2str(colIdx)));
            logMessage(sprintf('已生成 %s 模式 9点总览。', upper(modeName)));
        catch ME
            logError(ME);
        end
    end

%% --- Data loading / plotting ---
    function files = detectHistFiles(folder)
        files = blankFileMap();
        matFiles = dir(fullfile(folder, '*.mat'));

        for fileIdx = 1:numel(matFiles)
            fileName = matFiles(fileIdx).name;
            filePath = fullfile(folder, fileName);

            [pointIdx, phaseName] = parseHistFileName(fileName);
            if isempty(pointIdx) || pointIdx < 1 || pointIdx > 9
                continue;
            end

            if ~containsHistVariable(filePath)
                continue;
            end

            current = files(pointIdx).(phaseName);
            if isempty(current) || matFiles(fileIdx).datenum >= current.datenum
                files(pointIdx).(phaseName) = struct( ...
                    'path', filePath, ...
                    'name', fileName, ...
                    'datenum', matFiles(fileIdx).datenum);
            end
        end
    end

    function [pointIdx, phaseName] = parseHistFileName(fileName)
        pointIdx = [];
        phaseName = '';

        token = regexp(fileName, 'point(\d+).*_(obj|ref)\.mat$', 'tokens', 'once');
        if ~isempty(token)
            pointIdx = str2double(token{1});
            phaseName = token{2};
            return;
        end

        token = regexp(fileName, '^hist_(\d+)\.mat$', 'tokens', 'once');
        if ~isempty(token)
            pointIdx = str2double(token{1});
            phaseName = 'obj';
            return;
        end

        token = regexp(fileName, '^(\d+)\.mat$', 'tokens', 'once');
        if ~isempty(token)
            pointIdx = str2double(token{1});
            phaseName = 'obj';
        end
    end

    function tf = containsHistVariable(filePath)
        fileVars = whos('-file', filePath);
        varNames = {fileVars.name};
        tf = any(strcmp(varNames, 'hist')) || any(strcmp(varNames, 'histgram'));
        if ~tf && numel(fileVars) == 1
            tf = true;
        end
    end

    function updateModePopup()
        hasObj = any(arrayfun(@(s) ~isempty(s.obj), state.files));
        hasRef = any(arrayfun(@(s) ~isempty(s.ref), state.files));
        hasPair = any(arrayfun(@(s) ~isempty(s.obj) && ~isempty(s.ref), state.files));

        modeLabels = {};
        modeTags = {};

        if hasObj
            modeLabels{end + 1} = 'OBJ / RAW';
            modeTags{end + 1} = 'obj';
        end
        if hasRef
            modeLabels{end + 1} = 'REF';
            modeTags{end + 1} = 'ref';
        end
        if hasPair
            modeLabels{end + 1} = 'ABS(OBJ - REF)';
            modeTags{end + 1} = 'absdiff';
        end

        if isempty(modeLabels)
            modeLabels = {'OBJ'};
            modeTags = {'obj'};
        end

        set(handles.modePopup, 'String', modeLabels, 'Value', 1, 'UserData', modeTags);
    end

    function updateStatus()
        objCount = sum(arrayfun(@(s) ~isempty(s.obj), state.files));
        refCount = sum(arrayfun(@(s) ~isempty(s.ref), state.files));
        pairCount = sum(arrayfun(@(s) ~isempty(s.obj) && ~isempty(s.ref), state.files));

        if pairCount > 0
            set(handles.saveCalButton, 'Enable', 'on');
        else
            set(handles.saveCalButton, 'Enable', 'off');
        end

        if objCount > 0 || refCount > 0
            set(handles.intensityOverviewButton, 'Enable', 'on');
        else
            set(handles.intensityOverviewButton, 'Enable', 'off');
        end

        set(handles.statusText, 'String', sprintf( ...
            'OBJ/RAW: %d 个点 | REF: %d 个点 | 可 abs(obj-ref): %d 个点', ...
            objCount, refCount, pairCount));
    end

    function modeName = getSelectedMode()
        tags = get(handles.modePopup, 'UserData');
        value = get(handles.modePopup, 'Value');
        modeName = tags{value};
    end

    function [ny, nx] = getFirstAvailableHistSize(modeName)
        for pointIdx = 1:9
            [histCube, ~] = loadPointHist(pointIdx, modeName);
            if ~isempty(histCube)
                ny = size(histCube, 1);
                nx = size(histCube, 2);
                return;
            end
        end

        error('当前模式没有可加载的 hist 数据。');
    end

    function [histCube, timeAxis_ns, fileLabel] = loadPointHist(pointIdx, modeName)
        histCube = [];
        timeAxis_ns = [];
        fileLabel = '';

        switch modeName
            case 'obj'
                if isempty(state.files(pointIdx).obj)
                    return;
                end
                [histCube, timeAxis_ns] = loadHistFile(state.files(pointIdx).obj.path);
                fileLabel = state.files(pointIdx).obj.name;

            case 'ref'
                if isempty(state.files(pointIdx).ref)
                    return;
                end
                [histCube, timeAxis_ns] = loadHistFile(state.files(pointIdx).ref.path);
                fileLabel = state.files(pointIdx).ref.name;

            case 'absdiff'
                if isempty(state.files(pointIdx).obj) || isempty(state.files(pointIdx).ref)
                    return;
                end
                [objHist, timeAxis_ns] = loadHistFile(state.files(pointIdx).obj.path);
                [refHist, refTimeAxis] = loadHistFile(state.files(pointIdx).ref.path);
                [objHist, refHist] = cropToCommonSize(objHist, refHist);
                histCube = abs(double(objHist) - double(refHist));

                if isempty(timeAxis_ns)
                    timeAxis_ns = refTimeAxis;
                end
                if ~isempty(timeAxis_ns)
                    timeAxis_ns = timeAxis_ns(1:min(numel(timeAxis_ns), size(histCube, 3)));
                end
                fileLabel = sprintf('%s - %s', state.files(pointIdx).obj.name, state.files(pointIdx).ref.name);

            otherwise
                error('Unknown display mode: %s', modeName);
        end

        histCube = double(histCube);
        histCube(~isfinite(histCube)) = 0;
    end

    function [histCube, timeAxis_ns] = loadHistFile(filePath)
        data = load(filePath);
        timeAxis_ns = [];

        if isfield(data, 'hist')
            histCube = data.hist;
        elseif isfield(data, 'histgram')
            histCube = data.histgram;
        else
            fieldNames = fieldnames(data);
            histCube = data.(fieldNames{1});
        end

        if isfield(data, 'timeAxis_ns')
            timeAxis_ns = data.timeAxis_ns;
        end

        if ndims(histCube) ~= 3
            error('File does not contain a 3D hist cube: %s', filePath);
        end
    end

    function saveCalPointFile(pointIdx, calFilePath)
        objFile = state.files(pointIdx).obj.path;
        refFile = state.files(pointIdx).ref.path;

        objData = load(objFile);
        refData = load(refFile);

        objHistName = getHistVariableName(objData);
        refHistName = getHistVariableName(refData);
        objHist = objData.(objHistName);
        refHist = refData.(refHistName);

        if ndims(objHist) ~= 3 || ndims(refHist) ~= 3
            error('OBJ/REF must both contain 3D hist cubes for point %d.', pointIdx);
        end

        [objHist, refHist] = cropToCommonSize(objHist, refHist);
        calHist = abs(double(objHist) - double(refHist));
        calHist(~isfinite(calHist)) = 0;

        saveData = objData;
        saveData.(objHistName) = calHist;

        timeAxis_ns = chooseAxisData(objData, refData, 'timeAxis_ns', size(calHist, 3));
        if ~isempty(timeAxis_ns)
            saveData.timeAxis_ns = timeAxis_ns;
        end

        binEdges_ns = chooseAxisData(objData, refData, 'binEdges_ns', size(calHist, 3) + 1);
        if ~isempty(binEdges_ns)
            saveData.binEdges_ns = binEdges_ns;
        end

        if isfield(saveData, 'scanInfo') && isstruct(saveData.scanInfo) && isscalar(saveData.scanInfo)
            saveData.scanInfo.phase = 'cal';
            saveData.scanInfo.file = char(calFilePath);
        end

        save(calFilePath, '-struct', 'saveData');
    end

    function histName = getHistVariableName(data)
        if isfield(data, 'hist')
            histName = 'hist';
        elseif isfield(data, 'histgram')
            histName = 'histgram';
        else
            fieldNames = fieldnames(data);
            histName = fieldNames{1};
        end
    end

    function axisData = chooseAxisData(primaryData, fallbackData, axisName, maxLength)
        axisData = [];
        if isfield(primaryData, axisName)
            axisData = primaryData.(axisName);
        elseif isfield(fallbackData, axisName)
            axisData = fallbackData.(axisName);
        end

        if ~isempty(axisData) && isvector(axisData) && numel(axisData) > maxLength
            axisData = axisData(1:maxLength);
        end
    end

    function [a, b] = cropToCommonSize(a, b)
        commonSize = min(size(a), size(b));
        a = a(1:commonSize(1), 1:commonSize(2), 1:commonSize(3));
        b = b(1:commonSize(1), 1:commonSize(2), 1:commonSize(3));
    end

    function curve = sumSelectedCurve(histCube, rowIdx, colIdx)
        if max(rowIdx) > size(histCube, 1) || max(colIdx) > size(histCube, 2)
            error('选择的行/列索引超过 hist 尺寸：%d x %d。', size(histCube, 1), size(histCube, 2));
        end

        selectedCube = histCube(rowIdx, colIdx, :);
        curve = squeeze(sum(sum(selectedCube, 1), 2));
        curve = curve(:);
    end

    function [xAxis, xLabelText] = makeTimeAxis(curve, timeAxis_ns)
        if ~isempty(timeAxis_ns) && numel(timeAxis_ns) == numel(curve)
            xAxis = timeAxis_ns(:);
            xLabelText = 'Time (ns)';
        else
            xAxis = (1:numel(curve)).';
            xLabelText = 'Bin';
        end
    end

    function openFullPointView(pointIdx, modeName)
        try
            [histCube, ~, fileLabel] = loadPointHist(pointIdx, modeName);
            if isempty(histCube)
                logMessage(sprintf('点 %d 没有可打开的数据。', pointIdx));
                return;
            end

            my_display_hist(histCube);
            fig = gcf;
            set(fig, 'Name', sprintf('%d %s | %s | %s', ...
                pointIdx, pointNames{pointIdx}, upper(modeName), fileLabel), ...
                'NumberTitle', 'off');
            logMessage(sprintf('已打开点 %d (%s) 的完整 32x32 hist 视图。', pointIdx, pointNames{pointIdx}));
        catch ME
            logError(ME);
        end
    end

%% --- Helpers ---
    function files = blankFileMap()
        emptyFile = [];
        files = repmat(struct('obj', emptyFile, 'ref', emptyFile), 9, 1);
    end

    function tf = hasAnyData(files)
        tf = any(arrayfun(@(s) ~isempty(s.obj) || ~isempty(s.ref), files));
    end

    function tf = hasAnyPairData(files)
        tf = any(arrayfun(@(s) ~isempty(s.obj) && ~isempty(s.ref), files));
    end

    function calFileName = makeCalFileName(objFileName, pointIdx)
        calFileName = regexprep(objFileName, '_obj\.mat$', '_cal.mat', 'ignorecase');
        if strcmp(calFileName, objFileName)
            [~, baseName, extension] = fileparts(objFileName);
            calFileName = sprintf('%s_cal%s', baseName, extension);
        end
    end

    function idx = parseIndexList(text, maxValue, label)
        text = lower(strtrim(text));
        if isempty(text) || strcmp(text, ':') || strcmp(text, 'all')
            idx = 1:maxValue;
            return;
        end

        parts = regexp(text, '[,;\s]+', 'split');
        idx = [];

        for partIdx = 1:numel(parts)
            part = strtrim(parts{partIdx});
            if isempty(part)
                continue;
            end

            if contains(part, ':')
                nums = str2double(strsplit(part, ':'));
                if any(~isfinite(nums)) || ~(numel(nums) == 2 || numel(nums) == 3)
                    error('%s 格式错误：%s', label, part);
                end

                if numel(nums) == 2
                    idx = [idx, nums(1):nums(2)]; %#ok<AGROW>
                else
                    idx = [idx, nums(1):nums(2):nums(3)]; %#ok<AGROW>
                end
            else
                value = str2double(part);
                if ~isfinite(value)
                    error('%s 格式错误：%s', label, part);
                end
                idx(end + 1) = value; %#ok<AGROW>
            end
        end

        idx = unique(round(idx), 'stable');
        idx = idx(idx >= 1 & idx <= maxValue);

        if isempty(idx)
            error('%s 没有有效索引，允许范围是 1 到 %d。', label, maxValue);
        end
    end

    function logMessage(message)
        timestamped = sprintf('[%s] %s', datestr(now, 'HH:MM:SS'), message);
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
end
