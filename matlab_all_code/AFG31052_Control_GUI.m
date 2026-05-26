function AFG31052_Control_GUI
    % Tektronix AFG31052 simple dual-channel DC voltage controller.
    % Voltage means V(positive terminal) - V(negative/ground terminal).
    % Positive and negative values are both allowed.

    fig = uifigure('Name', 'Tektronix AFG31052 简单电压控制', ...
        'Position', [100, 100, 560, 300]);

    app.afgObj = [];

    %% ================== Connection ==================
    uilabel(fig, 'Text', 'VISA 地址:', 'Position', [20, 245, 80, 22]);
    app.visaEdit = uieditfield(fig, 'text', ...
        'Position', [100, 245, 310, 22], ...
        'Value', 'USB0::0x0699::0x035E::C018251::INSTR');

    app.connectBtn = uibutton(fig, 'Text', '连接仪器', ...
        'Position', [430, 241, 105, 30], ...
        'ButtonPushedFcn', @(btn, event) connectInstrument());

    uilabel(fig, ...
        'Text', '电压值 = 输出正端相对负端/地的电压，允许输入负数。', ...
        'Position', [20, 215, 500, 22], ...
        'FontColor', [0.25, 0.25, 0.25]);

    %% ================== CH1 / X ==================
    uilabel(fig, 'Text', '【 通道 X / CH1 】', ...
        'Position', [55, 170, 140, 22], 'FontWeight', 'bold');

    uilabel(fig, 'Text', '电压 (V):', 'Position', [35, 130, 70, 22]);
    app.voltageEditX = uieditfield(fig, 'numeric', ...
        'Position', [105, 130, 100, 22], ...
        'Value', 0.0);

    uilabel(fig, 'Text', '输出开关:', 'Position', [35, 85, 70, 22]);
    app.switchX = uiswitch(fig, 'rocker', ...
        'Items', {'OFF', 'ON'}, ...
        'Position', [125, 72, 40, 40], ...
        'Enable', 'off', ...
        'ValueChangedFcn', @(sw, event) toggleOutput(1, sw.Value));

    %% ================== CH2 / Y ==================
    uilabel(fig, 'Text', '【 通道 Y / CH2 】', ...
        'Position', [335, 170, 140, 22], 'FontWeight', 'bold');

    uilabel(fig, 'Text', '电压 (V):', 'Position', [315, 130, 70, 22]);
    app.voltageEditY = uieditfield(fig, 'numeric', ...
        'Position', [385, 130, 100, 22], ...
        'Value', 0.0);

    uilabel(fig, 'Text', '输出开关:', 'Position', [315, 85, 70, 22]);
    app.switchY = uiswitch(fig, 'rocker', ...
        'Items', {'OFF', 'ON'}, ...
        'Position', [405, 72, 40, 40], ...
        'Enable', 'off', ...
        'ValueChangedFcn', @(sw, event) toggleOutput(2, sw.Value));

    %% ================== Apply ==================
    app.setBothBtn = uibutton(fig, ...
        'Text', '同时设置 X / Y 输出电压', ...
        'Position', [155, 25, 250, 36], ...
        'Enable', 'off', ...
        'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(btn, event) setBothVoltages());

    %% ================== Callbacks ==================
    function connectInstrument()
        visaAddress = app.visaEdit.Value;
        try
            app.afgObj = visadev(visaAddress);
            configureTerminator(app.afgObj, "LF");
            app.afgObj.Timeout = 5;

            idn = scpiQuery(app.afgObj, '*IDN?');
            configureHighZLoad(1);
            configureHighZLoad(2);
            uialert(fig, ['成功连接到仪器：', idn], '连接成功', 'Icon', 'success');

            app.switchX.Enable = 'on';
            app.switchY.Enable = 'on';
            app.setBothBtn.Enable = 'on';
            app.connectBtn.BackgroundColor = [0.4, 0.8, 0.4];
            app.connectBtn.Text = '已连接';
        catch ME
            uialert(fig, ['连接失败：', ME.message], '错误', 'Icon', 'error');
        end
    end

    function setBothVoltages()
        if isempty(app.afgObj)
            return;
        end

        vX = app.voltageEditX.Value;
        vY = app.voltageEditY.Value;

        try
            setDcVoltage(1, vX);
            setDcVoltage(2, vY);
        catch ME
            uialert(fig, ['电压设置失败：', ME.message], '控制错误', 'Icon', 'error');
        end
    end

    function setDcVoltage(chanIn, voltageV)
        % DC mode output: Vout = offset.
        % Keep a tiny amplitude for instruments/firmware that require an amplitude value.
        configureHighZLoad(chanIn);
        scpiWrite(app.afgObj, sprintf('SOUR%d:FUNC:SHAP DC', chanIn));
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:UNIT VPP', chanIn));
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:AMPL 0.001VPP', chanIn));
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:OFFS %.9gV', chanIn, voltageV));
    end

    function configureHighZLoad(chanIn)
        % The AFG has a 50 ohm source. BNC-to-Dupont / DMM / scope high-Z loads
        % should use MAX load setting, otherwise measured open-circuit voltage
        % can be twice the programmed voltage.
        scpiWrite(app.afgObj, sprintf('OUTP%d:IMP MAX', chanIn));
    end

    function toggleOutput(chanIn, state)
        if isempty(app.afgObj)
            return;
        end

        try
            scpiWrite(app.afgObj, sprintf('OUTP%d:STAT %s', chanIn, state));
        catch ME
            uialert(fig, ['通道开关失败：', ME.message], '控制错误', 'Icon', 'error');
        end
    end

    fig.CloseRequestFcn = @(src, event) closeApp(src);
    function closeApp(src)
        if ~isempty(app.afgObj)
            try
                % Do not turn AFG outputs off when driving galvo analog inputs.
                % Output OFF can leave the galvo input floating, causing whine/jitter.
                setDcVoltage(1, app.voltageEditX.Value);
                setDcVoltage(2, app.voltageEditY.Value);
                scpiWrite(app.afgObj, 'OUTP1:STAT ON');
                scpiWrite(app.afgObj, 'OUTP2:STAT ON');
                clear app.afgObj;
            catch
            end
        end
        delete(src);
    end

    function response = scpiQuery(obj, cmd)
        response = strtrim(writeread(obj, cmd));
    end

    function scpiWrite(obj, cmd)
        writeline(obj, cmd);
    end
end
