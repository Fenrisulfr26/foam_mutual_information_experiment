function my_display_hist_gui()
%MY_DISPLAY_HIST_GUI Select a MAT file and display its histogram variable.
%
% Usage:
%   my_display_hist_gui
%
% The selected MAT file should contain a 3D histogram variable. If a
% variable named "hist" exists, it is selected automatically.

    selectedFile = "";

    fig = uifigure( ...
        'Name', 'Display Histogram From File', ...
        'Position', [320, 220, 720, 430]);

    grid = uigridlayout(fig, [7, 1]);
    grid.RowHeight = {44, 34, 34, 34, 34, '1x', 40};
    grid.Padding = [14, 14, 14, 14];
    grid.RowSpacing = 10;

    buttonGrid = uigridlayout(grid, [1, 3]);
    buttonGrid.ColumnWidth = {120, 120, '1x'};
    buttonGrid.Padding = [0, 0, 0, 0];
    buttonGrid.ColumnSpacing = 10;

    uibutton(buttonGrid, ...
        'Text', 'Select file', ...
        'ButtonPushedFcn', @selectFile);

    btnDisplay = uibutton(buttonGrid, ...
        'Text', 'Display', ...
        'Enable', 'off', ...
        'ButtonPushedFcn', @displaySelectedHist);

    lblStatus = uilabel(buttonGrid, ...
        'Text', 'No file selected', ...
        'HorizontalAlignment', 'right');

    lblFile = uilabel(grid, ...
        'Text', 'File:', ...
        'Interpreter', 'none');

    variableGrid = uigridlayout(grid, [1, 2]);
    variableGrid.ColumnWidth = {95, '1x'};
    variableGrid.Padding = [0, 0, 0, 0];

    uilabel(variableGrid, 'Text', 'Variable:');
    variableDropDown = uidropdown(variableGrid, ...
        'Items', {}, ...
        'Enable', 'off');

    imageModeGrid = uigridlayout(grid, [1, 2]);
    imageModeGrid.ColumnWidth = {95, '1x'};
    imageModeGrid.Padding = [0, 0, 0, 0];

    uilabel(imageModeGrid, 'Text', 'Left image:');
    imageModeDropDown = uidropdown(imageModeGrid, ...
        'Items', {'Total intensity', 'Peak intensity'}, ...
        'ItemsData', {'sum', 'peak'}, ...
        'Value', 'sum');

    optionGrid = uigridlayout(grid, [1, 3]);
    optionGrid.ColumnWidth = {220, 190, '1x'};
    optionGrid.Padding = [0, 0, 0, 0];
    optionGrid.ColumnSpacing = 10;

    chkDarkCorrected = uicheckbox(optionGrid, ...
        'Text', 'Apply dark correction', ...
        'Value', false);

    chkSmoothCurves = uicheckbox(optionGrid, ...
        'Text', 'Smooth all curves', ...
        'Value', false);

    smoothGrid = uigridlayout(optionGrid, [1, 2]);
    smoothGrid.ColumnWidth = {105, '1x'};
    smoothGrid.Padding = [0, 0, 0, 0];

    uilabel(smoothGrid, 'Text', 'Window bins:');
    smoothWindowField = uieditfield(smoothGrid, 'numeric', ...
        'Value', 5, ...
        'Limits', [1, Inf], ...
        'RoundFractionalValues', 'on');

    logArea = uitextarea(grid, ...
        'Editable', 'off', ...
        'Value', {'Select a MAT file that contains a 3D hist variable.'});

    uilabel(grid, ...
        'Text', 'The left 32x32 image and the right pixel curve use the same dark-correction and smoothing settings.', ...
        'FontColor', [0.25, 0.25, 0.25]);

    function selectFile(~, ~)
        [fileName, folderName] = uigetfile( ...
            {'*.mat', 'MAT-files (*.mat)'}, ...
            'Select a MAT file that contains hist');

        if isequal(fileName, 0)
            appendLog('File selection canceled.');
            return;
        end

        selectedFile = string(fullfile(folderName, fileName));
        lblFile.Text = ['File: ', char(selectedFile)];

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
                lblStatus.Text = 'No 3D variable';
                appendLog('No 3D histogram variable was found in this file.');
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
            lblStatus.Text = 'File selected';
            appendLog(sprintf('Found %d candidate 3D variable(s).', numel(candidates)));
        catch ME
            variableDropDown.Items = {};
            variableDropDown.Enable = 'off';
            btnDisplay.Enable = 'off';
            lblStatus.Text = 'Read failed';
            appendLog(ME.message);
        end
    end

    function displaySelectedHist(~, ~)
        if strlength(selectedFile) == 0
            appendLog('Select a file first.');
            return;
        end

        varName = variableDropDown.Value;
        try
            data = load(char(selectedFile), varName);
            histgram = data.(varName);
            darkText = 'off';

            if chkDarkCorrected.Value
                [histgram, correctionInfo] = correct_hot_dark_pixels(histgram);
                darkText = sprintf('on, %d pixels corrected', correctionInfo.numHotPixels);
            end

            displayOptions = struct();
            displayOptions.imageMode = imageModeDropDown.Value;
            displayOptions.smoothCurves = chkSmoothCurves.Value;
            displayOptions.smoothWindow = smoothWindowField.Value;

            my_display_hist(histgram, displayOptions);

            appendLog(sprintf(['Displayed %s, size %s, left image=%s, ', ...
                'smoothing=%s, dark correction=%s.'], ...
                varName, mat2str(size(histgram)), imageModeDropDown.Value, ...
                onOffText(chkSmoothCurves.Value), darkText));
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

function textValue = onOffText(flag)
    if flag
        textValue = 'on';
    else
        textValue = 'off';
    end
end
