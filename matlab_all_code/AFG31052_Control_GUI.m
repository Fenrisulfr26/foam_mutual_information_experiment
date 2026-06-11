function AFG31052_Control_GUI
    % Tektronix AFG31052 simple dual-channel DC voltage controller.
    % Voltage means V(positive terminal) - V(negative/ground terminal).
    % Positive and negative values are both allowed.

    fig = uifigure('Name', 'Tektronix AFG31052 简单电压控制', ...
        'Position', [100, 100, 720, 430]);

    app.afgObj = [];
    app.ch1Mode = 'dc';

    %% ================== Connection ==================
    uilabel(fig, 'Text', 'VISA 地址:', 'Position', [20, 365, 80, 22]);
    app.visaEdit = uieditfield(fig, 'text', ...
        'Position', [100, 365, 310, 22], ...
        'Value', 'USB0::0x0699::0x035E::C018251::INSTR');

    app.connectBtn = uibutton(fig, 'Text', '连接仪器', ...
        'Position', [430, 361, 105, 30], ...
        'ButtonPushedFcn', @(btn, event) connectInstrument());

    uilabel(fig, ...
        'Text', '电压值 = 输出正端相对负端/地的电压，允许输入负数。', ...
        'Position', [20, 335, 500, 22], ...
        'FontColor', [0.25, 0.25, 0.25]);

    %% ================== CH1 / X ==================
    uilabel(fig, 'Text', '【 通道 X / CH1 】', ...
        'Position', [55, 290, 140, 22], 'FontWeight', 'bold');

    uilabel(fig, 'Text', '电压 (V):', 'Position', [35, 250, 70, 22]);
    app.voltageEditX = uieditfield(fig, 'numeric', ...
        'Position', [105, 250, 100, 22], ...
        'Value', 0.0);

    uilabel(fig, 'Text', '输出开关:', 'Position', [35, 205, 70, 22]);
    app.switchX = uiswitch(fig, 'rocker', ...
        'Items', {'OFF', 'ON'}, ...
        'Position', [125, 192, 40, 40], ...
        'Enable', 'off', ...
        'ValueChangedFcn', @(sw, event) toggleOutput(1, sw.Value));

    %% ================== CH2 / Y ==================
    uilabel(fig, 'Text', '【 通道 Y / CH2 】', ...
        'Position', [335, 290, 140, 22], 'FontWeight', 'bold');

    uilabel(fig, 'Text', '电压 (V):', 'Position', [315, 250, 70, 22]);
    app.voltageEditY = uieditfield(fig, 'numeric', ...
        'Position', [385, 250, 100, 22], ...
        'Value', 0.0);

    uilabel(fig, 'Text', '输出开关:', 'Position', [315, 205, 70, 22]);
    app.switchY = uiswitch(fig, 'rocker', ...
        'Items', {'OFF', 'ON'}, ...
        'Position', [405, 192, 40, 40], ...
        'Enable', 'off', ...
        'ValueChangedFcn', @(sw, event) toggleOutput(2, sw.Value));

    %% ================== CH1 trapezoid waveform ==================
    uilabel(fig, 'Text', 'CH1 梯形波 / Trapezoid', ...
        'Position', [20, 158, 190, 22], 'FontWeight', 'bold');

    uilabel(fig, 'Text', '上升 (ns):', 'Position', [25, 125, 70, 22]);
    app.trapRiseNsEdit = uieditfield(fig, 'numeric', ...
        'Position', [95, 125, 70, 22], ...
        'Limits', [0.001, Inf], ...
        'Value', 5);

    uilabel(fig, 'Text', '保持 (ns):', 'Position', [180, 125, 70, 22]);
    app.trapHoldNsEdit = uieditfield(fig, 'numeric', ...
        'Position', [250, 125, 70, 22], ...
        'Limits', [0.001, Inf], ...
        'Value', 10);

    uilabel(fig, 'Text', '下降 (ns):', 'Position', [335, 125, 70, 22]);
    app.trapFallNsEdit = uieditfield(fig, 'numeric', ...
        'Position', [405, 125, 70, 22], ...
        'Limits', [0.001, Inf], ...
        'Value', 5);

    uilabel(fig, 'Text', '峰值 (V):', 'Position', [490, 125, 70, 22]);
    app.trapPeakVEdit = uieditfield(fig, 'numeric', ...
        'Position', [560, 125, 70, 22], ...
        'Value', 3.3);

    uilabel(fig, 'Text', '重频 (MHz):', 'Position', [25, 90, 70, 22]);
    app.trapFreqMHzEdit = uieditfield(fig, 'numeric', ...
        'Position', [95, 90, 70, 22], ...
        'Limits', [0.001, Inf], ...
        'Value', 20);

    uilabel(fig, 'Text', '低电平 (V):', 'Position', [180, 90, 70, 22]);
    app.trapLowVEdit = uieditfield(fig, 'numeric', ...
        'Position', [250, 90, 70, 22], ...
        'Value', 0);

    app.setTrapBtn = uibutton(fig, ...
        'Text', '设置 CH1 梯形波', ...
        'Position', [405, 84, 225, 34], ...
        'Enable', 'off', ...
        'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(btn, event) setCh1Trapezoid());

    %% ================== Apply ==================
    app.setBothBtn = uibutton(fig, ...
        'Text', '同时设置 X / Y 输出电压', ...
        'Position', [235, 25, 250, 36], ...
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
            app.setTrapBtn.Enable = 'on';
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
            app.ch1Mode = 'dc';
        catch ME
            uialert(fig, ['电压设置失败：', ME.message], '控制错误', 'Icon', 'error');
        end
    end

    function setCh1Trapezoid()
        if isempty(app.afgObj)
            return;
        end

        riseNs = app.trapRiseNsEdit.Value;
        holdNs = app.trapHoldNsEdit.Value;
        fallNs = app.trapFallNsEdit.Value;
        peakV = app.trapPeakVEdit.Value;
        lowV = app.trapLowVEdit.Value;
        freqMHz = app.trapFreqMHzEdit.Value;

        try
            configureCh1Trapezoid(riseNs, holdNs, fallNs, peakV, lowV, freqMHz);
            app.ch1Mode = 'trapezoid';
            app.switchX.Value = 'ON';
            uialert(fig, sprintf(['CH1 梯形波已设置：\n', ...
                '上升 = %.6g ns，保持 = %.6g ns，下降 = %.6g ns\n', ...
                '峰值 = %.6g V，低电平 = %.6g V，重频 = %.6g MHz'], ...
                riseNs, holdNs, fallNs, peakV, lowV, freqMHz), ...
                'CH1 梯形波', 'Icon', 'success');
        catch ME
            uialert(fig, ['CH1 梯形波设置失败：', ME.message], '控制错误', 'Icon', 'error');
        end
    end

    function setDcVoltage(chanIn, voltageV)
        % DC mode output: Vout = offset.
        % Keep a tiny amplitude for instruments/firmware that require an amplitude value.
        configureHighZLoad(chanIn);
        scpiWrite(app.afgObj, sprintf('SOUR%d:FUNC:SHAP DC', chanIn));
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:OFFS %.9gV', chanIn, voltageV));
    end

    function configureCh1Trapezoid(riseNs, holdNs, fallNs, peakV, lowV, freqMHz)
        chanIn = 1;
        periodS = 1 / (freqMHz * 1e6);
        riseS = riseNs * 1e-9;
        holdS = holdNs * 1e-9;
        fallS = fallNs * 1e-9;

        % Tektronix pulse width is specified at the 50% amplitude crossing.
        % This makes the flat-top duration close to the requested hold time.
        widthS = holdS + 0.5 * riseS + 0.5 * fallS;

        if peakV <= lowV
            error('峰值电压必须大于低电平。');
        end

        if riseS + holdS + fallS >= periodS
            error('上升时间 + 保持时间 + 下降时间必须小于一个周期。');
        end

        configure50OhmLoad(chanIn);
        setVoltageLimitsToInstrumentRange(chanIn);
        scpiWrite(app.afgObj, sprintf('SOUR%d:FUNC:SHAP PULS', chanIn));
        scpiWrite(app.afgObj, sprintf('SOUR%d:FREQ %.12gHz', chanIn, freqMHz * 1e6));
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:UNIT VPP', chanIn));
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:LEV:IMM:HIGH %.12gV', chanIn, peakV));
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:LEV:IMM:LOW %.12gV', chanIn, lowV));
        scpiWrite(app.afgObj, sprintf('SOUR%d:PULS:PER %.12gS', chanIn, periodS));
        scpiWrite(app.afgObj, sprintf('SOUR%d:PULS:WIDT %.12gS', chanIn, widthS));
        scpiWrite(app.afgObj, sprintf('SOUR%d:PULS:TRAN:LEAD %.12gS', chanIn, riseS));
        scpiWrite(app.afgObj, sprintf('SOUR%d:PULS:TRAN:TRA %.12gS', chanIn, fallS));
        scpiWrite(app.afgObj, sprintf('OUTP%d:STAT ON', chanIn));
    end

    function configureHighZLoad(chanIn)
        % The AFG has a 50 ohm source. BNC-to-Dupont / DMM / scope high-Z loads
        % should use MAX load setting, otherwise measured open-circuit voltage
        % can be twice the programmed voltage.
        scpiWrite(app.afgObj, sprintf('OUTP%d:IMP MAX', chanIn));
    end

    function configure50OhmLoad(chanIn)
        % Use 50 ohm load setting when the driven device is 50 ohm terminated.
        % Then programmed high/low levels match the voltage delivered to the load.
        scpiWrite(app.afgObj, sprintf('OUTP%d:IMP 50', chanIn));
    end

    function setVoltageLimitsToInstrumentRange(chanIn)
        % Let the AFG choose the valid limit range for its current load/model.
        % Disable concurrent voltage copy first so CH2 settings do not clamp CH1.
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:CONC:STAT OFF', chanIn));
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:LIM:LOW MIN', chanIn));
        scpiWrite(app.afgObj, sprintf('SOUR%d:VOLT:LIM:HIGH MAX', chanIn));
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
                if strcmp(app.ch1Mode, 'dc')
                    setDcVoltage(1, app.voltageEditX.Value);
                end
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
