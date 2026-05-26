function dark_pixel_correction_gui()
%DARK_PIXEL_CORRECTION_GUI GUI for batch dark-pixel correction of MAT files.
%
% The selected MAT files must contain a variable named "hist". Each output
% file is saved beside the source file with "_dark_corrected" appended to
% the original file name.

    selectedFiles = strings(0, 1);

    fig = uifigure( ...
        'Name', 'Dark Pixel Correction', ...
        'Position', [260, 180, 760, 500]);

    mainGrid = uigridlayout(fig, [4, 1]);
    mainGrid.RowHeight = {44, '1x', 130, 34};
    mainGrid.Padding = [14, 14, 14, 14];
    mainGrid.RowSpacing = 10;

    buttonGrid = uigridlayout(mainGrid, [1, 4]);
    buttonGrid.ColumnWidth = {130, 130, 130, '1x'};
    buttonGrid.Padding = [0, 0, 0, 0];
    buttonGrid.ColumnSpacing = 10;

    btnSelect = uibutton(buttonGrid, ...
        'Text', '选择文件', ...
        'ButtonPushedFcn', @selectFiles);

    btnProcess = uibutton(buttonGrid, ...
        'Text', '开始处理', ...
        'Enable', 'off', ...
        'ButtonPushedFcn', @processFiles);

    btnClear = uibutton(buttonGrid, ...
        'Text', '清空列表', ...
        'Enable', 'off', ...
        'ButtonPushedFcn', @clearFiles);

    lblStatus = uilabel(buttonGrid, ...
        'Text', '未选择文件', ...
        'HorizontalAlignment', 'right');

    fileList = uilistbox(mainGrid, ...
        'Items', {}, ...
        'Multiselect', 'on');

    logArea = uitextarea(mainGrid, ...
        'Editable', 'off', ...
        'Value', {'日志'});

    uilabel(mainGrid, ...
        'Text', '输出文件：原文件名 + _dark_corrected.mat；变量 hist 会被替换为修正后的 hist。', ...
        'FontColor', [0.25, 0.25, 0.25]);

    function selectFiles(~, ~)
        [names, pathName] = uigetfile( ...
            {'*.mat', 'MAT-files (*.mat)'}, ...
            '选择一个或多个含有 hist 的 MAT 文件', ...
            'MultiSelect', 'on');

        if isequal(names, 0)
            appendLog('已取消选择。');
            return;
        end

        if ischar(names)
            names = {names};
        end

        newFiles = strings(numel(names), 1);
        for i = 1:numel(names)
            newFiles(i) = string(fullfile(pathName, names{i}));
        end

        selectedFiles = unique([selectedFiles; newFiles], 'stable');
        refreshFileList();
        appendLog(sprintf('已选择 %d 个文件。', numel(selectedFiles)));
    end

    function processFiles(~, ~)
        if isempty(selectedFiles)
            appendLog('没有可处理的文件。');
            return;
        end

        btnSelect.Enable = 'off';
        btnProcess.Enable = 'off';
        btnClear.Enable = 'off';
        drawnow;

        nDone = 0;
        nFailed = 0;
        appendLog(sprintf('开始处理 %d 个文件...', numel(selectedFiles)));

        for i = 1:numel(selectedFiles)
            inFile = char(selectedFiles(i));
            try
                outFile = correct_dark_pixel_file(inFile);
                nDone = nDone + 1;
                appendLog(sprintf('[OK] %s', outFile));
            catch ME
                nFailed = nFailed + 1;
                appendLog(sprintf('[失败] %s', inFile));
                appendLog(sprintf('       %s', ME.message));
            end
            lblStatus.Text = sprintf('进度：%d / %d', i, numel(selectedFiles));
            drawnow;
        end

        appendLog(sprintf('完成：成功 %d 个，失败 %d 个。', nDone, nFailed));
        btnSelect.Enable = 'on';
        refreshFileList();
    end

    function clearFiles(~, ~)
        selectedFiles = strings(0, 1);
        refreshFileList();
        appendLog('文件列表已清空。');
    end

    function refreshFileList()
        fileList.Items = cellstr(selectedFiles);
        if isempty(selectedFiles)
            lblStatus.Text = '未选择文件';
            btnProcess.Enable = 'off';
            btnClear.Enable = 'off';
        else
            lblStatus.Text = sprintf('已选择 %d 个文件', numel(selectedFiles));
            btnProcess.Enable = 'on';
            btnClear.Enable = 'on';
        end
    end

    function appendLog(message)
        oldValue = logArea.Value;
        if ischar(oldValue)
            oldValue = {oldValue};
        end
        timestamp = string(datetime('now', 'Format', 'HH:mm:ss'));
        logArea.Value = [oldValue; {sprintf('[%s] %s', timestamp, message)}];
        scroll(logArea, 'bottom');
    end
end
