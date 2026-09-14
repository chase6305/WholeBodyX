# 开源参考与设计取舍

调研日期：2026-09-05。以下记录官方仓库、文档及查阅的接口文件；链接中的
`main` / `devel` 会随上游更新。它们是设计参考，当前运行依赖由 `uv.lock`
和 `conan.lock` 单独固定。本文的 WholeBodyX 取舍是本项目的设计决定。

| 项目 | 上游已有能力 | WholeBodyX 借鉴方式 |
| --- | --- | --- |
| [Pink](https://github.com/stephane-caron/pink) | Python 差分逆运动学，将任务与关节约束组成 QP，输出关节速度 | 先建立速度级 WBC；任务线性化与求解器分开，保留 Python 原生实现 |
| [TSID](https://github.com/stack-of-tasks/tsid) | C++ 逆动力学控制，提供任务、接触、分层 QP 及 Python 绑定 | 后续动力学 WBC 独立建模，明确加速度、接触力与驱动力矩的含义 |
| [Crocoddyl](https://github.com/loco-3d/crocoddyl) | C++ 最优控制，提供动力学、代价、约束、导数以及 Python 抽象接口 | 将优化问题、数值求解、计算工作区分开；复杂 MPC 后端通过 adapter 接入 |
| [OCS2](https://github.com/leggedrobotics/ocs2) | C++ 非线性最优控制与 MPC，提供参考管理和运行接口 | 参考带时间信息，规划与跟踪可不同频率，更新发生在明确的同步边界 |
| [acados](https://github.com/acados/acados) | 非线性最优控制求解器，Python 接口可生成和构建 C 求解代码 | 为未来 Python 建模、编译后部署保留接入方式 |
| [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | C++ 刚体运动学/动力学及 Python 接口 | 复用 URDF 和 Jacobian 计算，框架维护自己的状态及坐标约定 |

具体查阅与对应决策：

- Pink 的 [solve_ik.py](https://github.com/stephane-caron/pink/blob/main/pink/solve_ik.py)
  将问题构造与求解连接起来。本项目提供 `formulate()`，允许直接检查 QP，
  并通过 `QPSolver` 替换数值后端。没有复制 Pink 的任务或求解代码。
- TSID 的 [InverseDynamicsFormulationAccForce](https://github.com/stack-of-tasks/tsid/blob/devel/include/tsid/formulations/inverse-dynamics-formulation-acc-force.hpp)
  提供运动/力/驱动任务、接触和解码接口。本项目当前只实现速度变量的加权 QP；
  分层优先级与动力学约束属于后续独立功能。
- Crocoddyl 的 [ActionModelAbstract](https://github.com/loco-3d/crocoddyl/blob/devel/include/crocoddyl/core/action-base.hpp)
  将问题值、导数和数据工作区组织起来。本项目先实现可检查的关节积分模型；
  后续接入非线性问题时再引入完整导数接口。
- OCS2 的 [Optimal Control Modules](https://leggedrobotics.github.io/ocs2/optimal_control_modules.html)
  和 [MPC_MRT_Interface](https://github.com/leggedrobotics/ocs2/blob/main/ocs2_mpc/include/ocs2_mpc/MPC_MRT_Interface.h)
  是参考管理、缓冲及规划/执行边界的依据。本项目目前用确定性的单线程多频率示例验证数据链路，
  跨进程传输及异步 worker 尚未实现。
- acados 的 [Python Interface](https://docs.acados.org/python_interface/index.html)
  展示了建模、生成代码与运行求解器的接口。它是后续适配对象，当前没有加入运行依赖。
- Pinocchio 的 [官方介绍](https://stack-of-tasks.github.io/pinocchio/index.html)
  和 [逆运动学示例](https://github.com/stack-of-tasks/pinocchio/blob/devel/examples/inverse-kinematics.py)
  用于核实模型能力。WholeBodyX 的可选 Python adapter 对外返回数组，
  Jacobian 使用末端原点、世界方向的 `[linear; angular]` 约定，并用有限差分验证。

绑定和包管理另行选择：

- TSID 的 [README](https://github.com/stack-of-tasks/tsid#python-bindings) 与
  Crocoddyl 的 [README](https://github.com/loco-3d/crocoddyl) 说明它们使用
  Boost.Python / EigenPy；WholeBodyX 新扩展使用 pybind11。这些绑定系统之间
  不假设机器人模型对象能自动互通，适配边界使用数值数组与本项目的数据类型。
- [Conan 的 CMake 集成](https://docs.conan.io/2/integrations/cmake.html)
  支持先解析依赖、再通过工具链配置 CMake。C++ 库可独立构建。
- [Conan editable](https://docs.conan.io/2/tutorial/developing_packages/editable_packages.html)
  适合后续三个仓库的本地联调；正式验证消费打包后的库，使用锁文件与明确 profiles。
- [uv 的路径依赖](https://docs.astral.sh/uv/concepts/projects/dependencies/)
  适合 Python 算法仓库联调。
- [scikit-build-core](https://scikit-build-core.readthedocs.io/en/stable/guide/getting_started.html)
  用于构建可选 native wheel；[pybind11](https://pybind11.readthedocs.io/en/stable/advanced/misc.html)
  用于跨语言调用。释放 GIL 的入口只调用 C++，不包含 Python 回调。
- QP 统一为 OSQP 的标准形式。参考
  [C 0.6 API](https://osqp.org/docs/release-0.6.3/interfaces/C.html) 与
  [MPC 示例](https://osqp.org/docs/examples/mpc.html)，自行实现本项目的 WBC/MPC 问题构造。

当前 ConanCenter 查询到的 OSQP recipe 是 0.6.2 / 0.6.3，因此 C++ 固定 0.6.3；
Python 使用 1.x 并由 uv 锁定。两端 adapter 分别处理 API 差异，通过数学问题和
数值结果测试确认一致性；不交换 OSQP 内部对象。后续有需要时可维护 1.x Conan recipe。
