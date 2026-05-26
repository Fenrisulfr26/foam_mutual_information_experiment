%% AFG31052 scan 9 galvo points
% CH1 -> X voltage, CH2 -> Y voltage.
% Order: left-top -> middle-top -> right-top,
%        left-middle -> center -> right-middle,
%        left-bottom -> middle-bottom -> right-bottom.

clear; clc;

%% --- AFG connection ---
visaAddress = 'USB0::0x0699::0x035E::C018251::INSTR';
afg = [];

%% --- Scan parameters ---
dwellTimeS = 0.5;     % stay time at each point, seconds
useHighZLoad = true;  % use High-Z load for galvo controller analog input
parkXVoltageV = -0.090;  % park at center point before releasing MATLAB connection
parkYVoltageV =  0.025;

pointNames = {
    '左上'
    '中上'
    '右上'
    '左中'
    '中心'
    '右中'
    '左下'
    '中下'
    '右下'
};

scanPointsV = [
     0.300,  0.450;   % 1 左上
    -0.100,  0.450;   % 2 中上
    -0.510,  0.450;   % 3 右上
     0.300,  0.025;   % 4 左中
    -0.090,  0.025;   % 5 中心
    -0.510,  0.025;   % 6 右中
     0.300, -0.385;   % 7 左下
    -0.085, -0.385;   % 8 中下
    -0.500, -0.385;   % 9 右下
];

try
    afg = visadev(visaAddress);
    configureTerminator(afg, "LF");
    afg.Timeout = 5;

    idn = strtrim(writeread(afg, '*IDN?'));
    fprintf('Connected: %s\n', idn);

    configureChannelForDc(afg, 1, useHighZLoad);
    configureChannelForDc(afg, 2, useHighZLoad);

    writeline(afg, 'OUTP1:STAT ON');
    writeline(afg, 'OUTP2:STAT ON');

    for idx = 1:size(scanPointsV, 1)
        xV = scanPointsV(idx, 1);
        yV = scanPointsV(idx, 2);

        setDcVoltage(afg, 1, xV);
        setDcVoltage(afg, 2, yV);

        fprintf('Point %d/9 %s: X = %.3f V, Y = %.3f V\n', ...
            idx, pointNames{idx}, xV, yV);

        pause(dwellTimeS);
    end

    fprintf('9-point scan finished.\n');
    parkAndReleaseAfg(afg, parkXVoltageV, parkYVoltageV);
    afg = [];

catch ME
    fprintf('Error: %s\n', ME.message);
    parkAndReleaseAfg(afg, parkXVoltageV, parkYVoltageV);
    rethrow(ME);
end

%% --- Local functions ---
function configureChannelForDc(afg, chan, useHighZLoad)
    if useHighZLoad
        writeline(afg, sprintf('OUTP%d:IMP MAX', chan));
    end

    writeline(afg, sprintf('SOUR%d:FUNC:SHAP DC', chan));
    writeline(afg, sprintf('SOUR%d:VOLT:UNIT VPP', chan));
    writeline(afg, sprintf('SOUR%d:VOLT:AMPL 0.001VPP', chan));
end

function setDcVoltage(afg, chan, voltageV)
    writeline(afg, sprintf('SOUR%d:VOLT:OFFS %.9gV', chan, voltageV));
end

function parkAndReleaseAfg(afg, parkXVoltageV, parkYVoltageV)
    if isempty(afg)
        return;
    end

    try
        setDcVoltage(afg, 1, parkXVoltageV);
        setDcVoltage(afg, 2, parkYVoltageV);
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
