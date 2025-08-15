from tyxonq import Circuit, Param, waveforms

def create_precise_rabi_circuit(t_duration, amplitude, frequency):
    """
    创建精确的 Rabi 振荡实验电路
    
    参数:
    - t_duration: 脉冲持续时间（采样周期）
    - amplitude: 正弦波振幅 (|amp| ≤ 2)
    - frequency: 正弦波频率 (采样周期的倒数)
    """
    qc = Circuit(1)
    qc.use_pulse()
    
    # 创建参数化波形
    param_t = Param("t")
    
    sine_wave = waveforms.Sine(
        duration=t_duration,      # 持续时间
        amp=amplitude,            # 振幅
        frequency=frequency,           # 频率
    )
    
    # 构建校准程序
    builder = qc.calibrate("precise_rabi", [param_t])
    builder.new_frame("drive_frame", param_t)
    builder.play("drive_frame", sine_wave)
    builder.build()
    
    # 调用校准程序
    qc.add_calibration('precise_rabi', ['q[0]'])
    
    return qc

# 创建不同参数的电路进行参数扫描
frequencies = [0.01, 0.02, 0.05, 0.1]  # 不同频率
amplitudes = [0.5, 1.0, 1.5]            # 不同振幅

for freq in frequencies:
    for amp in amplitudes:
        qc = create_precise_rabi_circuit(
            t_duration=100,    # 100个采样周期
            amplitude=amp,      # 振幅
            frequency=freq      # 频率
        )
        print(f"Frequency: {freq}, Amplitude: {amp}")
        print(qc.to_tqasm())
        print("-" * 50)
