# WBC / MPC 可视化

控制开发通常需要同时查看几何状态、数值诊断和闭环仿真。下面按工具能力区分用途：

| 工具 | 用途 | WholeBodyX 当前选择 |
| --- | --- | --- |
| [Viser](https://viser.studio/main/) | 浏览器 3D 场景、机器人/轨迹显示、滑条与按钮 | 已接入，作为交互实验台 |
| [MeshCat](https://github.com/meshcat-dev/meshcat-python) | WebGL 机器人及几何可视化，可配合现有模型库 | 可替换的场景后端，当前未接入 |
| [MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html) | 动力学仿真及交互 viewer | 后续用于动力学/接触闭环，当前未接入 |
| [uPlot](https://viser.studio/main/examples/gui/uplot/) | Viser 内嵌时间序列曲线 | 已用于误差及求解耗时 |
| [Plotly](https://viser.studio/main/examples/gui/plotly_integration/) | 交互式数据分析图表 | 可用于后续离线报告，当前无运行依赖 |

这里的工具取舍是 WholeBodyX 的设计建议；3D 显示本身不提供动力学仿真。

## 启动

在仓库根目录执行：

```bash
uv sync --locked --extra viz
uv run --extra viz python -m wholebodyx.visualization
```

浏览器访问 `http://127.0.0.1:8080`。默认暂停，点击“运行”开始。
端口可通过 `--port` 指定，默认仅监听本机地址。远程使用时可建立 SSH 端口转发：

```bash
ssh -L 8080:127.0.0.1:8080 ubuntu@你的服务器
```

有 native 构建产物时可以直接使用 C++ WBC/MPC：

```bash
PYTHONPATH=build/native uv run --extra viz python -m wholebodyx.visualization --backend native
```

也可以安装生成的 native wheel 后在界面切换后端。扩展无法导入或缺少必要接口时，
菜单不提供 native；`wholebodyx doctor --require native` 会显示原因。
统一入口为 `wholebodyx view tracking`；Python 应用可传入自定义注册表，见 [接入说明](integration.md)。

## 已实现的显示与操作

- 青色、紫色机械臂显示实际关节状态，橙色线段和标记显示目标。
- 黄色折线显示 MPC 预测位置通过 FK 映射出的末端轨迹；细线显示实际末端历史。
- 六个关节目标滑条位于默认折叠的左右臂面板，修改目标后触发重新规划。
- 支持运行/暂停、单步、重置，以及 WBC / MPC + WBC 控制链路切换。
- 同一界面可选 Python 或 C++ 算法后端。
- “WBC 后端”和“MPC 后端”独立选择，支持混合运行。
  纯 WBC 模式禁用 MPC 菜单，并且不初始化 MPC 实例。
- “示例场景”可切换 `tracking`、`cartesian_circle`、`coordination`、`mpc_retargeting`。
  另有 `disturbance_recovery`，用于验证关节状态偏移后的预测偏差检测。
  切换场景会重置时钟/曲线，并采用相应任务配方；详细算法见 [示例指南](../examples/README.md)。
- 显示左右末端位置误差、WBC/MPC 求解耗时、迭代数和最大约束违反。
  MPC 只在实际重规划时记录求解耗时，未求解的周期保留缺测值。
- 默认保存最近 300 个诊断采样点，曲线约每 0.2 s 仿真时间更新。
- 重规划时间诊断同样最多保留 300 条，避免长时间运行后持续增长。
- 暂停时修改目标会立即刷新当前误差并清除过期预测；重置与故障会清除排队的单步操作。
- 定时换目标的显示边界同样立即隐藏旧预测；求解失败后旧计划被清除。
- 暂停保留最新求解和参考诊断，故障信息保留到成功执行或重置。
- 显示参考更新原因、计划剩余时间，以及状态偏移/预测偏差重规划的累计次数。

圆轨迹和协作场景用橙色标记显示即时末端目标，没有对应的关节目标机器人轮廓。
其他场景同时显示名义关节目标。MPC 预测的 FK 在每次重新规划时计算并缓存。
无界面示例不再计算显示用的预测 FK；viewer 在新计划首次显示时生成缓存。

当前展示的是两条平面 3R 链，使用速度积分仿真。URDF 数值 adapter 已有独立测试，
真实机器人 mesh 加载、动力学仿真、接触力箭头和碰撞显示尚未加入这个 viewer。

## 与控制层的关系

包内 `wholebodyx.session.TrackingSession` 不依赖 Viser，负责模型、任务和仿真步骤。
旧的 `wholebodyx.visualization.session` 保留兼容导入。
`ControlViewer` 读取 session 状态并更新画面。GUI 回调只排队保存操作意图，
主循环在仿真步骤边界应用目标、模式与重置操作；模型和求解器不在回调线程中运行。

`visualization` 是 Python 可选模块。C++ 程序和基础 Python 控制包都不要求安装 Viser。
未来接真实运行时，建议把可视化消费的数据限制为状态、目标、预测和诊断快照，
以较低频率刷新，避免渲染反向阻塞控制循环。
