%% scan points

% 对应 Python minimalmodbus 版本
% 功能：向 0x0050 和 0x0051 两个保持寄存器写入 X/Y 电压值
% 注意：MATLAB 的 Modbus 地址是 1-based，会自动减 1
% 如果设备手册写的是 0x0050，MATLAB 中通常要写 0x0050 + 1

clear; clc;

% --- 配置参数 ---
PORT = "COM6";        % Windows: COMx
SLAVE_ID = 1;         % Modbus 从
BAUDRATE = 9600;      % 波特率

% --- 扫描点电压：从左上开始逐点扫描 ---
% 顺序：上排左到右 -> 中排左到右 -> 下排左到右
scan_point_names = [
    "左上"
    "上中"
    "右上"
    "左中"
    "中心"
    "右中"
    "左下"
    "下中"
    "右下"
];

scan_points_mv = [
     340,  450;   % 1 左上
     -55,  455;   % 2 上中
    -470,  460;   % 3 右上
     340,   30;   % 4 左中
     -50,   30;   % 5 中心
    -460,   30;   % 6 右中
     350, -390;   % 7 左下
     -55, -390;   % 8 下中
    -460, -390;   % 9 右下
];

SCAN_DWELL_S = 1;   % 每个扫描点停留时间，单位 s

% --- 安全保护 ---
% 你之前说只允许 +-1 V，这里按 +-1000 mV 限制
V_LIMIT_MV = 1000;

if any(abs(scan_points_mv(:)) > V_LIMIT_MV)
    error("目标电压超出安全范围：只允许 %.0f mV 到 %.0f mV", ...
        -V_LIMIT_MV, V_LIMIT_MV);
end

% --- 根据设备协议转换写入值 ---
% 你的 Python 逻辑是：
% write_value = target_voltage + 30000
%
% 例如：
% target_voltage = 0 mV    -> 30000
% target_voltage = 1000 mV -> 31000
% target_voltage = -1000 mV -> 29000

% --- 寄存器地址 ---
% Python 里写的是：
% 0x0050 和 0x0051
%
% 注意：MATLAB 的 modbus write 使用 1-based addressing；
% 文档说明 MATLAB 会对输入地址自动减 1。
% 所以如果设备手册地址是 0x0050，MATLAB 这里建议写 0x0050 + 1。

REG_X = hex2dec("0050") + 1;
REG_Y = hex2dec("0051") + 1;

% --- 初始化 Modbus RTU ---
try
    m = modbus("serialrtu", PORT, "Timeout", 1);

    % 设置串口参数
    m.BaudRate = BAUDRATE;
    m.DataBits = 8;
    m.Parity   = "none";
    m.StopBits = 1;

    fprintf("成功连接到 %s，准备发送指令...\n", PORT);

    % --- 逐点扫描并写入寄存器 ---
    % write(m, 'holdingregs', address, value, serverId, precision)
    % precision 默认是 uint16，这里显式指定为 uint16

    num_points = size(scan_points_mv, 1);
    for idx = 1:num_points
        target_voltage_x = scan_points_mv(idx, 1);
        target_voltage_y = scan_points_mv(idx, 2);

        write_value_x = uint16(target_voltage_x + 30000);
        write_value_y = uint16(target_voltage_y + 30000);

        write(m, "holdingregs", REG_X, double(write_value_x), SLAVE_ID, "uint16");
        write(m, "holdingregs", REG_Y, double(write_value_y), SLAVE_ID, "uint16");

        fprintf("扫描点 %d/%d（%s）：X = %d mV，写入值: %d；Y = %d mV，写入值: %d\n", ...
            idx, num_points, char(scan_point_names(idx)), ...
            target_voltage_x, write_value_x, target_voltage_y, write_value_y);

        if idx < num_points && SCAN_DWELL_S > 0
            pause(SCAN_DWELL_S);
        end
    end

    % --- 可选：读取回传值 ---
    % 如果模块支持读取保持寄存器，可以取消下面注释：
    %
    % read_x = read(m, "holdingregs", REG_X, 1, SLAVE_ID, "uint16");
    % read_y = read(m, "holdingregs", REG_Y, 1, SLAVE_ID, "uint16");
    %
    % fprintf("模块反馈 X 写入值: %d，对应电压: %d mV\n", read_x, read_x - 30000);
    % fprintf("模块反馈 Y 写入值: %d，对应电压: %d mV\n", read_y, read_y - 30000);

    % --- 关闭连接 ---
    clear m;

catch ME
    fprintf("发生错误：%s\n", ME.message);

    fprintf("\n请检查：\n");
    fprintf("1. COM 口是否正确，例如 COM6 是否存在。\n");
    fprintf("2. RS485 A/B 线是否接反。\n");
    fprintf("3. 模块是否已经供电。\n");
    fprintf("4. 从站地址 SLAVE_ID 是否正确。\n");
    fprintf("5. 波特率、校验位、停止位是否与设备手册一致。\n");
    fprintf("6. MATLAB 是否安装 Industrial Communication Toolbox。\n");

    if exist("m", "var")
        clear m;
    end
end

%% single point set


% 对应 Python minimalmodbus 版本
% 功能：向 0x0050 和 0x0051 两个保持寄存器写入 X/Y 电压值
% 注意：MATLAB 的 Modbus 地址是 1-based，会自动减 1
% 如果设备手册写的是 0x0050，MATLAB 中通常要写 0x0050 + 1

clear; clc;

% --- 配置参数 ---
PORT = "COM6";        % Windows: COMx
SLAVE_ID = 1;         % Modbus 从
BAUDRATE = 9600;      % 波特率

% --- 扫描点电压：从左上开始逐点扫描 ---
% 顺序：上排左到右 -> 中排左到右 -> 下排左到右
scan_point_names = [
    "左上"
    "上中"
    "右上"
    "左中"
    "中心"
    "右中"
    "左下"
    "下中"
    "右下"
];

scan_points_mv = [
     340,  450;   % 1 左上
     -55,  455;   % 2 上中
    -470,  460;   % 3 右上
     340,   30;   % 4 左中
     -50,   30;   % 5 中心
    -460,   30;   % 6 右中
     350, -390;   % 7 左下
     -55, -390;   % 8 下中
    -460, -390;   % 9 右下
];

SCAN_DWELL_S = 1;   % 每个扫描点停留时间，单位 s

% --- 安全保护 ---
% 你之前说只允许 +-1 V，这里按 +-1000 mV 限制
V_LIMIT_MV = 1000;

if any(abs(scan_points_mv(:)) > V_LIMIT_MV)
    error("目标电压超出安全范围：只允许 %.0f mV 到 %.0f mV", ...
        -V_LIMIT_MV, V_LIMIT_MV);
end

% --- 根据设备协议转换写入值 ---
% 你的 Python 逻辑是：
% write_value = target_voltage + 30000
%
% 例如：
% target_voltage = 0 mV    -> 30000
% target_voltage = 1000 mV -> 31000
% target_voltage = -1000 mV -> 29000

% --- 寄存器地址 ---
% Python 里写的是：
% 0x0050 和 0x0051
%
% 注意：MATLAB 的 modbus write 使用 1-based addressing；
% 文档说明 MATLAB 会对输入地址自动减 1。
% 所以如果设备手册地址是 0x0050，MATLAB 这里建议写 0x0050 + 1。

REG_X = hex2dec("0050") + 1;
REG_Y = hex2dec("0051") + 1;

% --- 初始化 Modbus RTU ---
try
    m = modbus("serialrtu", PORT, "Timeout", 1);

    % 设置串口参数
    m.BaudRate = BAUDRATE;
    m.DataBits = 8;
    m.Parity   = "none";
    m.StopBits = 1;

    fprintf("成功连接到 %s，准备发送指令...\n", PORT);

    % --- 逐点扫描并写入寄存器 ---
    % write(m, 'holdingregs', address, value, serverId, precision)
    % precision 默认是 uint16，这里显式指定为 uint16

    num_points = size(scan_points_mv, 1);
    for idx = 5
        target_voltage_x = scan_points_mv(idx, 1);
        target_voltage_y = scan_points_mv(idx, 2);

        write_value_x = uint16(target_voltage_x + 30000);
        write_value_y = uint16(target_voltage_y + 30000);

        write(m, "holdingregs", REG_X, double(write_value_x), SLAVE_ID, "uint16");
        write(m, "holdingregs", REG_Y, double(write_value_y), SLAVE_ID, "uint16");

        fprintf("扫描点 %d/%d（%s）：X = %d mV，写入值: %d；Y = %d mV，写入值: %d\n", ...
            idx, num_points, char(scan_point_names(idx)), ...
            target_voltage_x, write_value_x, target_voltage_y, write_value_y);

        if idx < num_points && SCAN_DWELL_S > 0
            pause(SCAN_DWELL_S);
        end
    end

    % --- 可选：读取回传值 ---
    % 如果模块支持读取保持寄存器，可以取消下面注释：
    %
    % read_x = read(m, "holdingregs", REG_X, 1, SLAVE_ID, "uint16");
    % read_y = read(m, "holdingregs", REG_Y, 1, SLAVE_ID, "uint16");
    %
    % fprintf("模块反馈 X 写入值: %d，对应电压: %d mV\n", read_x, read_x - 30000);
    % fprintf("模块反馈 Y 写入值: %d，对应电压: %d mV\n", read_y, read_y - 30000);

    % --- 关闭连接 ---
    clear m;

catch ME
    fprintf("发生错误：%s\n", ME.message);

    fprintf("\n请检查：\n");
    fprintf("1. COM 口是否正确，例如 COM6 是否存在。\n");
    fprintf("2. RS485 A/B 线是否接反。\n");
    fprintf("3. 模块是否已经供电。\n");
    fprintf("4. 从站地址 SLAVE_ID 是否正确。\n");
    fprintf("5. 波特率、校验位、停止位是否与设备手册一致。\n");
    fprintf("6. MATLAB 是否安装 Industrial Communication Toolbox。\n");

    if exist("m", "var")
        clear m;
    end
end

