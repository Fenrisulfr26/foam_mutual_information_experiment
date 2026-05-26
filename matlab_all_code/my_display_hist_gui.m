function my_display_hist_gui()
%MY_DISPLAY_HIST_GUI Select a MAT file and display its hist variable.
%
% Usage:
%   my_display_hist_gui
%
% The selected MAT file should contain a 3D histogram variable. If a
% variable named "hist" exists, it is selected automatically.

    fig = uifigure( ...
        'Name', 'Display Histogram From File', ...
        'Position', [320, 260, 680, 360]);

    grid = uigridlayout(fig, [5, 1]);
    grid.RowHeight = {44, 34, 34, '1x', 40};
    grid.Padding = [14, 14, 14, 14];
    grid.RowSpacing = 10;

    buttonGrid = uigridlayout(grid, [1, 3]);
    buttonGrid.ColumnWidth = {120, 120, '1x'};
    buttonGrid.Padding = [0, 0, 0, 0];
    buttonGrid.ColumnSpacing = 10;

    uibutton(buttonGrid, ...
        'Text', '选择文件', ...
        'ButtonPushedFcn', @selectFile);

    btnDisplay = uibutton(buttonGrid, ...
        'Text', '展示', ...
        'Enable', 'off', ...
        'ButtonPushedFcn', @displaySelectedHist);

    lblStatus = uilabel(buttonGrid, ...
        'Text', '未选择文件', ...
        'HorizontalAlignment', 'right');

    lblFile = uilabel(grid, ...
        'Text', '文件：', ...
        'Interpreter', 'none');

    variableGrid = uigridlayout(grid, [1, 2]);
    variableGrid.ColumnWidth = {80, '1x'};
    variableGrid.Padding = [0, 0, 0, 0];

    uilabel(variableGrid, 'Text', '变量：');
    variableDropDown = uidropdown(variableGrid, ...
        'Items', {}, ...
        'Enable', 'off');

    logArea = uitextarea(grid, ...
        'Editable', 'off', ...
        'Value', {'请选择保存有 hist 的 MAT 文件。'});

    uilabel(grid, ...
        'Text', '展示窗口仍使用 my_display_hist：左侧累计强度图，右侧鼠标所在像素的时间直方图。', ...
        'FontColor', [0.25, 0.25, 0.25]);

    selectedFile = "";

    function selectFile(~, ~)
        [fileName, folderName] = uigetfile( ...
            {'*.mat', 'MAT-files (*.mat)'}, ...
            '选择保存有 hist 的 MAT 文件');

        if isequal(fileName, 0)
            appendLog('已取消选择。');
            return;
        end

        selectedFile = string(fullfile(folderName, fileName));
        lblFile.Text = ['文件：', char(selectedFile)];

        try
            vars = whos('-file', char(selectedFile));
            candidates = {};

            for i = 1:numel(vars)
                if numel(vars(i).size) == 3
                    candidates{end + 1} = vars(i).name; %#ok<AGROW>
                end
            end

            if isempty(candidates)
                variableDropDown.Items = {};
                variableDropDown.Enable = 'off';
                btnDisplay.Enable = 'off';
                lblStatus.Text = '没有 3D 变量';
                appendLog('这个文件里没有找到 3D histogram 变量。');
                return;
            end

            variableDropDown.Items = candidates;
            variableDropDown.Enable = 'on';

            if any(strcmp(candidates, 'hist'))
                variableDropDown.Value = 'hist';
            else
                variableDropDown.Value = candidates{1};
            end

            btnDisplay.Enable = 'on';
            lblStatus.Text = '已选择文件';
            appendLog(sprintf('找到 %d 个 3D 变量。', numel(candidates)));
        catch ME
            variableDropDown.Items = {};
            variableDropDown.Enable = 'off';
            btnDisplay.Enable = 'off';
            lblStatus.Text = '读取失败';
            appendLog(ME.message);
        end
    end

    function displaySelectedHist(~, ~)
        if strlength(selectedFile) == 0
            appendLog('请先选择文件。');
            return;
        end

        varName = variableDropDown.Value;
        try
            data = load(char(selectedFile), varName);
            histgram = data.(varName);
            my_display_hist(histgram);
            appendLog(sprintf('已展示变量 %s，尺寸 %s。', varName, mat2str(size(histgram))));
        catch ME
            appendLog(ME.message);
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
