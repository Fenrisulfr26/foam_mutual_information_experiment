import pmcx
import numpy as np

def run_multi_source_demo():
    print("==================================================================")
    # 1. 定义一个简单的 10x10x10 的均匀散射介质（全部标签为 1）
    vol = np.ones((10, 10, 10), dtype=np.uint8)

    # 2. 定义光学属性 (介质 0 为背景，介质 1 为散射介质)
    # prop: [mua, mus, g, n]
    prop = [
        [0.0, 0.0, 1.0, 1.0],      # 背景 (空气)
        [0.01, 1.0, 0.9, 1.37]     # 散射介质
    ]

    # 3. 定义两个独立的光源 (在 z=0 边界处，分别位于 x=3.0 和 x=7.0，入射方向均为 +z)
    srcpos = [
        [3.0, 5.0, 0.0],  # 光源 1
        [7.0, 5.0, 0.0]   # 光源 2
    ]
    srcdir = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ]

    # 4. 定义两个探测器 (在相对的 z=10 边界处，分别对齐光源 1 和 光源 2，半径为 2.0)
    detpos = [
        [3.0, 5.0, 10.0, 2.0],  # 探测器 1 (对齐光源 1)
        [7.0, 5.0, 10.0, 2.0]   # 探测器 2 (对齐光源 2)
    ]

    # 5. 配置 PMCX 参数
    cfg = {
        "nphoton": 200000,          # 总光子数 (每个光源各分配 100,000)
        "vol": vol,
        "unitinmm": 1.0,            # 空间单位：1mm/voxel
        "issrcfrom0": 1,
        "prop": prop,
        "srcpos": srcpos,
        "srcdir": srcdir,
        "srctype": "pencil",
        "detpos": detpos,
        "tstart": 0.0,
        "tend": 5e-9,               # 最大飞行时间 5 ns
        "tstep": 5e-9,              # 1个时间门
        "seed": 123456789,
        "gpuid": 1,
        "autopilot": 1,
        "issavedet": 1,
        "savedetflag": "dp",        # 保存探测器 ID (d) 和 偏路径 (p)
        "srcid": -1                 # 重要：-1 表示独立模拟多个光源，返回各自的光源 ID
    }

    print("Running PMCX multi-source simulation (1 simulation)...")
    res = pmcx.mcxlab(cfg)
    
    detp = res.get("detp")
    if not detp:
        print("Error: No detected photons found in simulation output.")
        return

    # 6. 从输出结果中提取 detector IDs 和 source IDs
    # PMCX 内部会自动将大整数的 raw detid 拆分为 1-based 的 detid 和 1-based 的 srcid
    detids = detp.get("detid")
    srcids = detp.get("srcid")

    if srcids is None:
        print("Error: 'srcid' not found in detected photons. Verify that 'srcid': -1 is used.")
        return

    print("Successfully retrieved detected photon history.")
    print(f"Total detected photons: {len(detids)}")

    # 7. 计算并输出光子来源与探测器的关联矩阵
    # 统计：光源 i 发射的光子被探测器 j 接收的数量
    # 探测器 ID: 1, 2
    # 光源 ID: 1, 2
    counts = np.zeros((2, 2), dtype=int)
    for s_id, d_id in zip(srcids, detids):
        # 转换为 0-based 索引
        s_idx = s_id - 1
        d_idx = d_id - 1
        if 0 <= s_idx < 2 and 0 <= d_idx < 2:
            counts[s_idx, d_idx] += 1

    print("\n================== Verification Matrix ==================")
    print("              Detector 1      Detector 2")
    print(f"Source 1      {counts[0, 0]:<15}{counts[0, 1]:<15}")
    print(f"Source 2      {counts[1, 0]:<15}{counts[1, 1]:<15}")
    print("=========================================================")

    # 8. 物理自洽性验证
    # 探测器 1 (x=3) 在物理上离光源 1 (x=3) 较近，离光源 2 (x=7) 较远；
    # 探测器 2 (x=7) 在物理上离光源 2 (x=7) 较近，离光源 1 (x=3) 较远。
    # 即使存在散射，直对的光源对该探测器的贡献也应当占主导地位。
    ratio1 = counts[0, 0] / max(1, counts[1, 0])
    ratio2 = counts[1, 1] / max(1, counts[0, 1])

    print(f"Detector 1 checks: photons from Source 1 / Source 2 = {ratio1:.2f}x")
    print(f"Detector 2 checks: photons from Source 2 / Source 1 = {ratio2:.2f}x")

    if counts[0, 0] > counts[1, 0] and counts[1, 1] > counts[0, 1]:
        print("\n[SUCCESS] Verification SUCCESS: Photons were successfully distinguished by source origin!")
        print("  - Detector 1 correctly received significantly more photons originating from Source 1.")
        print("  - Detector 2 correctly received significantly more photons originating from Source 2.")
    else:
        print("\n[FAILED] Verification FAILED: Unexpected distribution of source-detector counts.")

if __name__ == "__main__":
    run_multi_source_demo()
