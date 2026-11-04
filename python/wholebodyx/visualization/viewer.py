"""Viser scene and uPlot telemetry for the fixed-base dual-arm demo."""

import threading
import time
from operator import itemgetter

import numpy as np

from ..experiments import SCENARIOS
from ..session import TrackingSession


_PLOT_FIELDS = tuple(itemgetter(name) for name in
                     ("time", "left_error", "right_error", "wbc_ms", "mpc_ms"))


def plot_columns(history):
    """Build contiguous float columns without temporary Python lists.

    NumPy float conversion maps missing (None) timing samples to NaN.
    """
    return tuple(np.fromiter(map(field, history), dtype=float, count=len(history))
                 for field in _PLOT_FIELDS)


def segments(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) < 2:
        return np.zeros((1, 2, 3), dtype=np.float32)
    return np.stack((points[:-1], points[1:]), axis=1)


def link_orientations(points):
    """Cylinder local +z follows each planar link: Rz(angle + pi/2) Rx(pi/2)."""
    directions = np.diff(points, axis=0)
    half_angle = (np.arctan2(directions[:, 1], directions[:, 0]) + np.pi / 2) / 2
    cosine, sine = np.cos(half_angle) / np.sqrt(2), np.sin(half_angle) / np.sqrt(2)
    return np.column_stack((cosine, cosine, sine, sine))  # wxyz


class ControlViewer:
    """UI callbacks enqueue intent; the calling thread owns all solver state."""

    def __init__(self, host="127.0.0.1", port=8080, backend="python", mode="MPC + WBC",
                 scenario="tracking", mpc_backend=None, *, registry=None):
        import trimesh
        import viser
        import viser.uplot

        self.uplot = viser.uplot
        self.session = TrackingSession(backend, mode, scenario, mpc_backend, registry=registry)
        statuses = self.session.registry.status()
        controllers = tuple(status.name for status in statuses
                            if status.available and status.controller)
        planners = tuple(status.name for status in statuses if status.available and status.planner)
        if backend not in controllers or self.session.mpc_backend not in planners:
            failures = "; ".join(f"{status.name}: {status.detail}"
                                 for status in statuses if not status.available)
            raise ValueError(f"selected backend is unavailable in the viewer: {failures}")
        self._lock = threading.Lock()
        self._pending = {"reset": False, "steps": 0, "configuration": None, "goal": None,
                         "scenario": None, "refresh": False}
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid(
            "/ground", width=3.0, height=3.0, plane="xy", position=(0, 0, -0.06))
        self.server.scene.add_frame("/origin", axes_length=0.15, axes_radius=0.005)
        self.server.gui.configure_theme(
            dark_mode=True, control_layout="collapsible", control_width="large",
            show_logo=False, show_share_button=False)
        self.server.gui.add_markdown("# WholeBodyX\n双臂控制实验台 · 2 × 3R\n\n拖动视图旋转，滚轮缩放。")
        self.running = self.server.gui.add_checkbox("运行", initial_value=False)
        self.scenario = self.server.gui.add_dropdown(
            "示例场景", tuple(SCENARIOS), initial_value=scenario)
        self.description = self.server.gui.add_markdown(SCENARIOS[scenario].description)
        self.mode = self.server.gui.add_dropdown(
            "控制链路", ("WBC", "MPC + WBC"), initial_value=self.session.mode,
            disabled=SCENARIOS[scenario].mode is not None)
        self.backend = self.server.gui.add_dropdown("WBC 后端", controllers, initial_value=backend)
        self.mpc_backend = self.server.gui.add_dropdown(
            "MPC 后端", planners, initial_value=self.session.mpc_backend,
            disabled=self.session.mode == "WBC")
        reset = self.server.gui.add_button("重置机器人与曲线")
        step = self.server.gui.add_button("单步")
        self.status = self.server.gui.add_markdown("**已暂停** · 点击运行开始，或使用单步。")
        self._last_error = None
        self.sliders = []
        for side, offset in (("左臂关节目标 / rad", 0), ("右臂关节目标 / rad", 3)):
            with self.server.gui.add_folder(side, expand_by_default=False):
                for j in range(3):
                    slider = self.server.gui.add_slider(
                        f"关节 {j + 1}", min=-1.5, max=1.5, step=0.01,
                        initial_value=float(self.session.goal[offset + j]))
                    self.sliders.append(slider)
        self.server.gui.add_markdown("青色 / 紫色：实际机械臂　橙色：目标\n\n黄色：MPC 预测　细线：末端历史轨迹")
        with self.server.gui.add_folder("末端位置误差", expand_by_default=True):
            self.error_plot = self._plot("误差 / m", ("左臂", "右臂"))
        with self.server.gui.add_folder("求解耗时", expand_by_default=True):
            self.timing_plot = self._plot("耗时 / ms", ("WBC", "MPC"))

        def queue_reset(_):
            with self._lock:
                self._pending["reset"] = True
                self._pending["steps"] = 0

        def queue_step(_):
            with self._lock:
                self._pending["steps"] += 1

        def queue_configuration(_):
            with self._lock:
                mode = SCENARIOS[self.scenario.value].mode or self.mode.value
                self._pending["configuration"] = (self.backend.value, mode, self.mpc_backend.value)

        def queue_scenario(_):
            with self._lock:
                self._pending["scenario"] = self.scenario.value
                self._pending["steps"] = 0
                self._pending["goal"] = None

        def queue_refresh(_):
            with self._lock:
                self._pending["refresh"] = True

        def queue_goal(_):
            with self._lock:
                self._pending["goal"] = [slider.value for slider in self.sliders]

        reset.on_click(queue_reset)
        step.on_click(queue_step)
        self.mode.on_update(queue_configuration)
        self.backend.on_update(queue_configuration)
        self.mpc_backend.on_update(queue_configuration)
        self.scenario.on_update(queue_scenario)
        self.running.on_update(queue_refresh)
        for slider in self.sliders:
            slider.on_update(queue_goal)

        @self.server.on_client_connect
        def camera(client):
            client.camera.position = (1.5, -1.3, 1.45)
            client.camera.look_at = (0.4, 0.0, 0.0)

        self.arms = []
        colors = ((45, 212, 191), (167, 139, 250))
        for side, color in zip(("left", "right"), colors):
            links, joints = [], []
            for j, length in enumerate(self.session.model.lengths):
                mesh = trimesh.creation.cylinder(radius=0.025, height=length, sections=16)
                links.append(self.server.scene.add_mesh_simple(
                    f"/{side}/link_{j}", mesh.vertices, mesh.faces,
                    color=color, flat_shading=False))
            for j in range(4):
                joints.append(self.server.scene.add_icosphere(
                    f"/{side}/joint_{j}", radius=0.038, color=color))
            target = self.server.scene.add_icosphere(
                f"/{side}/target", radius=0.024, color=(251, 146, 60))
            goal_links = self.server.scene.add_line_segments(
                f"/{side}/goal", segments(np.zeros((4, 3))),
                colors=(251, 146, 60), thickness=2, thickness_units="screen")
            prediction = self.server.scene.add_line_segments(
                f"/{side}/prediction", segments([]),
                colors=(250, 204, 21), thickness=4, thickness_units="screen")
            trail = self.server.scene.add_line_segments(
                f"/{side}/trail", segments([]), colors=color, thickness=2, thickness_units="screen")
            self.arms.append((links, joints, target, goal_links, prediction, trail))
        self._last_plot = -1.0
        self._displayed_plans = [None] * len(self.arms)
        self._displayed_goals = [None] * len(self.arms)
        self.render()

    def _render_goal(self, index, frame, q, handle):
        session = self.session
        handle.visible = session.scenario in ("tracking", "mpc_retargeting", "disturbance_recovery")
        if not handle.visible:
            self._displayed_goals[index] = None
            return
        # Retain value snapshots: goal arrays and model lengths can change in
        # place. Include the model itself so replacing it invalidates geometry.
        key = (session.model, q.tobytes(), session.model.lengths.tobytes())
        if key != self._displayed_goals[index]:
            handle.points = segments(session.links(q, frame))
            self._displayed_goals[index] = key

    def _render_prediction(self, index, frame, handle):
        # Always check validity, including paused target edits and clock jumps.
        path = self.session.prediction(frame)
        plan = self.session.plan
        handle.visible = len(path) > 1
        if plan is not self._displayed_plans[index]:
            if len(path) > 1:
                handle.points = segments(path)
            self._displayed_plans[index] = plan

    def _plot(self, ylabel, names):
        return self.server.gui.add_uplot(
            # Viser 1.1 preserves relative zoom; an empty x-domain poisons it
            # with infinite bounds. A timestamp plus missing samples is valid.
            data=(np.array([0.0]), np.array([np.nan]), np.array([np.nan])),
            series=(self.uplot.Series(label="时间 / s"),
                    self.uplot.Series(label=names[0], stroke="#2dd4bf", width=2),
                    self.uplot.Series(label=names[1], stroke="#a78bfa", width=2)),
            scales={"x": self.uplot.Scale(time=False, auto=True)},
            axes=(self.uplot.Axis(label="仿真时间 / s"), self.uplot.Axis(label=ylabel)),
            legend=self.uplot.Legend(show=True), height=185)

    def render(self):
        session = self.session
        state = session.plant.read()
        reference = session.reference()
        plot_due = state.stamp - self._last_plot >= 0.2 or self._last_plot < 0
        errors = []
        with self.server.atomic():
            for k, frame in enumerate(session.frames):
                points = session.links(state.q, frame)
                if plot_due:
                    # The drawn chain already contains the end-effector
                    # position. Reuse it for telemetry at this same state.
                    errors.append(np.linalg.norm(points[-1] - reference.poses[k].translation))
                links, joints, target, goal_links, prediction, trail = self.arms[k]
                orientations = link_orientations(points)
                for j, handle in enumerate(links):
                    handle.wxyz = orientations[j]
                    handle.position = (points[j] + points[j + 1]) / 2
                for j, handle in enumerate(joints):
                    handle.position = points[j]
                target.position = reference.poses[k].translation
                self._render_goal(k, frame, reference.q, goal_links)
                self._render_prediction(k, frame, prediction)
                trail.points = segments(session.trails[k])
        if plot_due:
            history = list(session.history)
            if not history:
                history = [{"time": state.stamp, "left_error": errors[0], "right_error": errors[1],
                            "wbc_ms": None, "mpc_ms": None}]
            else:
                # A paused target edit updates the current error at the same timestamp.
                history[-1] = dict(history[-1], left_error=errors[0], right_error=errors[1])
            times, left, right, wbc, mpc = plot_columns(history)
            self.error_plot.data = (times, left, right)
            self.timing_plot.data = (times, wbc, mpc)
            self._last_plot = state.stamp

    def _show_status(self, sample):
        state_label = "运行中" if self.running.value else "已暂停"
        content = (f"**{state_label}** · {self.session.mode}\n\n"
            f"WBC {self.session.backend} / MPC {self.session.mpc_backend}\n\n"
            f"时间 {sample['time']:.2f} s · WBC {sample['wbc_ms']:.2f} ms\n\n"
            f"最大约束违反 {sample['violation']:.1e} · 迭代 {sample['iterations']}")
        if self.session.mode == "MPC + WBC":
            reasons = {"initial": "首次规划", "periodic": "周期更新", "reused": "沿用计划",
                       "goal_changed": "目标变化", "tracking_error": "预测偏差", "expired": "计划过期",
                       "coverage": "有效期不足", "retry": "故障后重试", "forced": "重新规划"}
            content += f"\n\n参考：{reasons.get(sample['replan_reason'], sample['replan_reason'])}"
            if sample["plan_remaining_s"] is not None and self.session.plan is not None:
                content += f" · 剩余 {sample['plan_remaining_s']:.2f} s"
            else:
                content += " · 等待新计划"
            if self.session.disturbances:
                content += (f"\n\n状态偏移 {self.session.disturbances} 次"
                                        f" · 偏差重规划 {self.session.tracking_replans} 次")

        self.status.content = content

    def tick(self):
        with self._lock:
            pending = self._pending.copy()
            self._pending["reset"] = False
            self._pending["configuration"] = None
            self._pending["goal"] = None
            self._pending["scenario"] = None
            self._pending["refresh"] = False
            if pending["reset"] or pending["scenario"] is not None:
                self._pending["steps"] = 0
            elif pending["steps"]:
                self._pending["steps"] -= 1
        if not (self.running.value or pending["steps"] or pending["reset"]
                or pending["configuration"] is not None or pending["goal"] is not None
                or pending["scenario"] is not None or pending["refresh"]):
            return
        try:
            reset = pending["reset"] or pending["scenario"] is not None
            if pending["scenario"] is not None:
                self.session.set_scenario(pending["scenario"])
                self.mode.value = self.session.mode
                self.mode.disabled = SCENARIOS[self.session.scenario].mode is not None
                self.description.content = SCENARIOS[self.session.scenario].description
                for slider, value in zip(self.sliders, self.session.goal):
                    slider.value = float(value)
            if pending["configuration"] is not None:
                backend, mode, mpc_backend = pending["configuration"]
                self.session.configure(
                    backend, SCENARIOS[self.session.scenario].mode or mode, mpc_backend)
            self.mpc_backend.disabled = self.session.mode == "WBC"
            if pending["goal"] is not None:
                self.session.set_goal(pending["goal"])
            if pending["reset"]:
                self.session.reset()
            if reset:
                self._last_error = None
                self._last_plot = -1.0
                self.status.content = "**已重置** · 时间 0.00 s · 当前关节目标保留。"
            elif self.running.value or pending["steps"]:
                sample = self.session.step()
                self._last_error = None
                self._show_status(sample)
            elif (pending["refresh"] or pending["goal"] is not None
                    or pending["configuration"] is not None):
                if not self._last_error:
                    if self.session.history:
                        self._show_status(self.session.history[-1])
                    else:
                        self.status.content = (
                            f"**已暂停** · 时间 {self.session.plant.read().stamp:.2f} s")
            if (any(pending[key] is not None for key in ("goal", "configuration", "scenario"))
                    or pending["refresh"]):
                self._last_plot = -1.0
            self.render()
        except (ValueError, TypeError, RuntimeError, ImportError) as exc:
            self._last_error = str(exc)
            self.running.value = False
            with self._lock:
                self._pending["steps"] = 0
                self._pending["refresh"] = False
            self.status.content = f"**已暂停**\n\n{exc}\n\n可调整目标或重置后继续。"
            self.render()  # A rejected plan must disappear from the browser immediately.

    def run(self):
        try:
            while True:
                started = time.monotonic()
                self.tick()
                time.sleep(max(0.001, self.session.dt - (time.monotonic() - started)))
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()
