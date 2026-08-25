# DM4310 + GM6020 PWM视觉云台

本工程由 `dm4310_gm6020_pwm_demo` 完整复制后修改。原工程没有改动。

## 1. 当前硬件分工

- DM4310：俯仰轴，FDCAN1 MIT位置控制，命令频率1kHz。
- GM6020：方位轴，PWM位置模式，PE9/TIM1_CH1输出50Hz PWM。
- 视觉串口：USART1，PA10接收、PA9发送预留，115200 8N1。
- 统一调度：TIM3以1kHz运行云台控制，GM PWM目标每20次更新一次。

视觉模块接线：

```text
RK3588 UART_TX -> STM32 PA10 / USART1_RX
RK3588 GND     -- STM32 GND
```

只允许3.3V TTL直接连接。不要连接RS-232电平，也不要使用串口排针给另一块主板供电。

## 2. 控制方法

视觉协议中的yaw和elevation是相机光轴误差，不是电机绝对角度。程序在每次TIM3中断中执行：

```text
目标角速度 = 视觉角度误差 * VISION_xxx_RATE_KP
目标位置  += 目标角速度 * 0.001s
```

因此视觉误差从30度下降到20度时，电机仍向同一方向运动，但速度会下降；误差到0后停止更新目标位置并保持；误差变为负数后反向修正。

默认状态机：

- `GIMBAL_TRACKING`：有效视觉误差控制两个轴。
- `GIMBAL_HOLD`：超过250ms没有新有效测量，保持两个轴最后目标。
- `GIMBAL_SEARCHING`：超过1000ms没有有效目标，GM以10度/秒在-150度到+150度之间往返搜索，DM保持俯仰。

收到新的有效目标后立即退出搜索。`SEARCHING`、`TEMP_LOST`、`FAULT`、CRC错误和过期测量不会被当成零误差。

## 3. GM6020调试助手设置

烧录和上电前确认：

- 电机类型：GM6020。
- PWM模式：位置模式。
- 最大转角：360度。
- 中点位置：按实际机械中位设置，当前建议4096。
- PWM行程已经校准，程序按1080-1920us有效行程使用。
- 第一次联调先限制较低转速，并保持机械结构可随时断电。

1500us对应调试助手配置的中点，不是每次开机位置。启动PWM后GM可能主动转向该中点。

## 4. 常用参数位置

主要参数都在 `User/gimbal_control.h`：

- `PITCH_MOTOR_DIRECTION`：DM俯仰方向，反向时改成 `-1.0f`。
- `YAW_MOTOR_DIRECTION`：GM方位方向，反向时改成 `-1.0f`。
- `VISION_PITCH_RATE_KP`、`VISION_YAW_RATE_KP`：视觉误差转换成调整速度的比例。
- `PITCH_MAX_TRACK_RATE_DEG_S`、`YAW_MAX_TRACK_RATE_DEG_S`：跟踪速度上限。
- `VISION_ERROR_DEADBAND_DEG`：中心死区。
- `VISION_ERROR_FILTER_ALPHA`：新视觉误差的低通滤波系数。
- `PITCH_MIN/MAX_ANGLE_DEG`：俯仰软件限位。
- `YAW_MIN/MAX_ANGLE_DEG`：方位命令限位，避免跨越正负180度边界。
- `VISION_VALID_HOLD_TIMEOUT_MS`：旧误差最长使用时间，当前250ms。
- `VISION_SEARCH_START_TIMEOUT_MS`：进入搜索前等待时间，当前1000ms。
- `GM_SEARCH_RATE_DEG_S`：目标丢失后的搜索速度。
- `DM_MIT_KP`、`DM_MIT_KD`：DM MIT控制参数。

改动这些宏以后必须重新编译并烧录。

## 5. 新增代码文件

- `User/vision_protocol.c/.h`：解析32字节视觉帧、CRC-16/CCITT-FALSE校验、半帧和粘包重新同步。
- `User/vision_uart.c/.h`：USART1逐字节中断接收、256字节环形缓冲、在主循环中送入协议解析器。
- `User/gimbal_control.c/.h`：数据有效性判断、方向映射、误差滤波、位置目标积分、限速、限位和失目标搜索。
- `Docs/communication-protocol.md`：用户提供的视觉通信协议副本，没有把其中内容当作程序操作指令。

旧的 `User/motion_demo.c/.h` 仍保留用于对照，但已经从Keil编译列表和 `main.c` 运行入口移除，不会控制电机。

## 6. 联调观察变量

可在Keil Watch中观察：

- `vision_uart_stats`：串口字节数、缓冲溢出和UART错误。
- `vision_protocol_stats`：正确帧、CRC错误帧和格式错误帧数量。
- `gimbal_telemetry`：当前状态、视觉误差、两个轴目标、GM脉宽、数据年龄和接收统计。

推荐先不连接电机，用协议固定发送 `yaw=+0.1rad`、`elevation=-0.05rad` 检查解码和方向，再连接单轴低速测试。
