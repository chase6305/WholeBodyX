"""H1 floating-base Pinocchio dynamics: standing and double-support CoM shifts."""
import argparse
import csv
import json
from pathlib import Path
import time
import threading
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from wholebodyx.dynamics import ContactWBC, DynamicsState, FloatingBaseDynamics
from wholebodyx.centroidal import CoMPreviewMPC

DEFAULT_URDF = Path(__file__).resolve().parents[3] / "HumanoidAssets/UnitreeH1/robot.urdf"


class DynamicsDemo:
    dt = .005
    duration = 8.

    def __init__(self, urdf=DEFAULT_URDF, motion="shift", amplitude=.035, controller="wbc", push_force=0.):
        if motion not in ("stand", "shift"):
            raise ValueError("motion must be stand or shift")
        if not np.isfinite(amplitude) or not 0 <= amplitude <= .05:
            raise ValueError("amplitude must be between 0 and 0.05 metres")
        if controller not in ("wbc", "mpc"):
            raise ValueError("controller must be wbc or mpc")
        if not np.isfinite(push_force) or not 0 <= push_force <= 5:
            raise ValueError("push_force must be in [0, 5] newtons")
        self.push_force = float(push_force)
        self.controller_mode = controller
        self.urdf, self.motion, self.amplitude = Path(urdf), motion, amplitude
        self.robot = FloatingBaseDynamics(self.urdf)
        if self.robot.nv != 25:
            raise ValueError("expected the 19-joint H1 with a floating base")
        self.reset()

    def reset(self):
        r = self.robot
        q = r.pin.neutral(r.model)
        for side in ("left", "right"):
            for joint, value in (("hip_yaw", .08 if side == "left" else -.08), ("hip_pitch", -.15), ("knee", .3), ("ankle", -.15)):
                q[r.model.joints[r.model.getJointId(f"{side}_{joint}_joint")].idx_q] = value
        state = DynamicsState(q, np.zeros(r.nv))
        q[2] -= min(p.translation[2] for p in r.terms(state)[4])
        self.state = DynamicsState(q, np.zeros(r.nv))
        self.controller = ContactWBC(r, self.state)
        if self.push_force:
            self.controller.contact_reserve = .5
        self.initial_com = r.terms(self.state)[5].copy()
        self.max_dynamics_residual = self.max_contact_error = self.max_violation = 0.
        self.steps = 0
        self._manual_push_until = 0.
        self._manual_push_force = 0.
        self.com_acceleration = np.zeros(3)
        self.last_mpc_ms = 0.
        self.mpc_continuity_fallbacks = 0
        self.planner = None
        if self.controller_mode == "mpc":
            support = np.array([(pose.translation + pose.rotation @ [x, y, 0.])[:2]
                                for pose in self.controller.anchors for x in (-.10, .10) for y in (-.035, .035)])
            self.planner = CoMPreviewMPC(support, self.initial_com[2])

    def target(self, stamp):
        target = self.initial_com.copy()
        if self.motion == "shift":
            target[1] += self.amplitude * np.sin(2 * np.pi * stamp / self.duration) ** 3
        return target

    def target_derivatives(self, stamp):
        velocity, acceleration = np.zeros(3), np.zeros(3)
        if self.motion == "shift":
            omega = 2 * np.pi / self.duration
            sine, cosine = np.sin(omega * stamp), np.cos(omega * stamp)
            velocity[1] = 3 * self.amplitude * omega * sine**2 * cosine
            acceleration[1] = 3 * self.amplitude * omega**2 * (2 * sine * cosine**2 - sine**3)
        return velocity, acceleration

    def trigger_push(self, force=3., duration=.2):
        if not np.isfinite([force, duration]).all() or not 0 <= force <= 5 or not 0 < duration <= .5:
            raise ValueError("push must be 0–5 N for at most 0.5 s")
        self.controller.contact_reserve = .5
        self._manual_push_force = float(force)
        self._manual_push_until = self.state.stamp + duration

    def _replan_com(self, terms, targets, target_velocities):
        """Keep reference acceleration continuous while feeding back measured p/v.

        Acceleration is the preview's command state, not a measured state to
        overwrite every 100 ms. If it is incompatible with measured p/v and
        support constraints, retry once from the plant acceleration instead.
        Both plans go through the same MPC feasibility checks.
        """
        planner = self.planner
        previous = planner.last_plan
        measured = self.com_acceleration[:2]
        seed = measured if previous is None else previous.sample(self.state.stamp)[2]

        def solve(acceleration):
            return planner.plan(terms[5][:2], terms[6][:2], acceleration, targets, self.state.stamp,
                                target_velocities=target_velocities)

        try:
            return solve(seed)
        except RuntimeError:
            if previous is None:
                raise
            self.mpc_continuity_fallbacks += 1
            return solve(measured)

    def step(self):
        target = self.target(self.state.stamp)
        velocity, acceleration = self.target_derivatives(self.state.stamp)
        mpc_ms = 0.
        if self.planner is not None:
            if self.steps % 20 == 0:
                before = time.perf_counter()
                terms = self.robot.terms(self.state)
                targets = np.array([self.target(self.state.stamp + (k + 1) * self.planner.dt)[:2]
                                    for k in range(self.planner.horizon)])
                target_velocities = np.array([self.target_derivatives(self.state.stamp + (k + 1) * self.planner.dt)[0][:2]
                                              for k in range(self.planner.horizon)])
                self._replan_com(terms, targets, target_velocities)
                mpc_ms = (time.perf_counter() - before) * 1000
                self.last_mpc_ms = mpc_ms
            target[:2], velocity[:2], acceleration[:2] = self.planner.last_plan.sample(self.state.stamp)
        started = time.perf_counter()
        command = self.controller.solve(self.state, target, self.dt,
                                        target_velocity=velocity, target_acceleration=acceleration)
        solve_ms = (time.perf_counter() - started) * 1000
        # Apply a world-Y force at the pelvis origin only in the plant. WBC
        # receives measured state on the next step, not the disturbance itself.
        phase = (self.steps * self.dt) % self.duration
        push = self.push_force if 2. <= phase < 2.2 else 0.
        if self.state.stamp < self._manual_push_until - 1e-10:
            push = max(push, self._manual_push_force)
        external_force = np.zeros(self.robot.nv)
        external_force[:3] = Rotation.from_quat(self.state.q[3:7]).as_matrix().T @ [0., push, 0.]
        acceleration, wrench = self.robot.forward(self.state, command["torque"], command["contact_rhs"], command["terms"],
                                                   external_force=external_force)
        mass, bias, jac = command["terms"][:3]
        forward_residual = float(np.max(np.abs(mass @ acceleration + bias -
            self.robot.selection @ command["torque"] - jac.T @ wrench - external_force)))
        agreement = float(np.max(np.abs(acceleration - command["acceleration"])))
        contact_violation = self.controller.contact_violation(wrench)
        if contact_violation > 1e-5:
            raise RuntimeError(f"forward contact wrench violates support bounds: {contact_violation}")
        self.state = self.robot.integrate(self.state, acceleration, self.dt)
        terms = self.robot.terms(self.state)
        self.steps += 1
        self.com_acceleration = terms[8] @ acceleration + terms[7]
        error = max(np.linalg.norm(p.translation - a.translation) for p, a in zip(terms[4], self.controller.anchors))
        self.max_contact_error = max(self.max_contact_error, float(error))
        self.max_dynamics_residual = max(self.max_dynamics_residual, command["dynamics_residual"])
        self.max_violation = max(self.max_violation, command["violation"])
        row = dict(time=self.state.stamp, com_y_m=float(terms[5][1]), target_com_y_m=float(self.target(self.state.stamp)[1]),
                   com_error_m=float(np.linalg.norm(terms[5] - self.target(self.state.stamp))),
                   forward_contact_violation=contact_violation, wbc_ms=solve_ms,
                   reference_com_y_m=float(target[1]), mpc_ms=mpc_ms, last_mpc_ms=self.last_mpc_ms, applied_push_y_n=float(push),
                   forward_dynamics_residual=forward_residual,
                   mpc_replans=0 if self.planner is None else self.planner.replans,
                   mpc_continuity_fallbacks=self.mpc_continuity_fallbacks,
                   solver_recoveries=getattr(self.controller, "solver_recoveries", 0),
                   left_fz_n=float(wrench[2]), right_fz_n=float(wrench[8]),
                   max_torque_nm=float(np.max(np.abs(command["torque"]))),
                   dynamics_residual=command["dynamics_residual"], contact_residual=command["contact_residual"],
                   contact_position_error_m=float(error), forward_acceleration_error=agreement,
                   constraint_violation=command["violation"])
        for i, value in enumerate(command["torque"]):
            row[f"tau_{i}_nm"] = float(value)
        return row


def run(urdf=DEFAULT_URDF, motion="shift", amplitude=.035, steps=1600, csv_path=None, controller="wbc", push_force=0.):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    demo = DynamicsDemo(urdf, motion, amplitude, controller, push_force)
    stream = None
    com_min, com_max, min_force = float("inf"), -float("inf"), float("inf")
    max_torque = max_forward_error = max_forward_contact = max_com_error = solve_total = 0.
    max_forward_residual = push_impulse = 0.
    try:
        if csv_path is not None:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("w", newline="")
        writer = None
        for _ in range(steps):
            row = demo.step()
            com_min, com_max = min(com_min, row["com_y_m"]), max(com_max, row["com_y_m"])
            min_force = min(min_force, row["left_fz_n"], row["right_fz_n"])
            max_torque = max(max_torque, row["max_torque_nm"])
            max_forward_error = max(max_forward_error, row["forward_acceleration_error"])
            max_forward_contact = max(max_forward_contact, row["forward_contact_violation"])
            max_com_error = max(max_com_error, row["com_error_m"])
            solve_total += row["wbc_ms"]
            max_forward_residual = max(max_forward_residual, row["forward_dynamics_residual"])
            push_impulse += row["applied_push_y_n"] * demo.dt
            if stream is not None:
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
    finally:
        if stream is not None:
            stream.close()
    return dict(motion=motion, periodic_push_force_n=push_force, floating_base=True, physics="prescribed_double_support_dynamics",
                dynamic_mpc=controller == "mpc", mpc_model="fixed_height_com_zmp" if controller == "mpc" else None,
                mpc_replans=row["mpc_replans"], solver_recoveries=row["solver_recoveries"], steps=steps, duration_s=row["time"],
                mpc_continuity_fallbacks=row["mpc_continuity_fallbacks"],
                mass_kg=float(demo.robot.pin.computeTotalMass(demo.robot.model)),
                com_y_range_m=com_max - com_min, min_normal_force_n=min_force,
                max_torque_nm=max_torque, max_com_error_m=max_com_error,
                mean_wbc_ms=solve_total / steps,
                max_dynamics_residual=demo.max_dynamics_residual,
                max_contact_position_error_m=demo.max_contact_error,
                max_constraint_violation=demo.max_violation,
                max_forward_contact_violation=max_forward_contact, applied_push_impulse_ns=push_impulse,
                max_forward_dynamics_residual=max_forward_residual,
                max_forward_acceleration_error=max_forward_error)


def view(urdf, motion, amplitude, host, port, autoplay=True, loop=True, controller="wbc", push_force=0.):
    import trimesh
    import viser
    demo = DynamicsDemo(urdf, motion, amplitude, controller, push_force)
    server = viser.ViserServer(host=host, port=port)
    try:
        server.scene.set_up_direction("+z")
        server.scene.add_grid("/ground", width=4, height=4)
        mode_description = "恒定高度 CoM/ZMP 预览 MPC → 力矩 WBC" if controller == "mpc" else "解析质心目标 → 力矩 WBC"
        server.gui.add_markdown(f"# H1 · Pinocchio 动力学\n{mode_description}\n\n"
                                "浮动基座、双脚固定接触。尚无接触切换、碰撞或落脚冲击。")
        running = server.gui.add_checkbox("运行", initial_value=autoplay)
        looping = server.gui.add_checkbox("循环播放", initial_value=loop)
        speed = server.gui.add_slider("播放速度", min=.25, max=2., step=.25, initial_value=1.)
        reset_requested, step_requested, push_requested = threading.Event(), threading.Event(), threading.Event()
        server.gui.add_button("复位并暂停").on_click(lambda _: reset_requested.set())
        server.gui.add_button("单步 5 ms").on_click(lambda _: step_requested.set())
        server.gui.add_button("侧向轻推 3 N · 0.2 s").on_click(lambda _: push_requested.set())
        status = server.gui.add_markdown("正在初始化")
        visuals = []
        for link in ET.parse(urdf).getroot().findall("link"):
            for index, visual in enumerate(link.findall("visual")):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                geometry = trimesh.load(Path(urdf).parent / mesh.attrib["filename"], force="mesh")
                geometry.apply_scale(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "))
                origin = visual.find("origin")
                xyz = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                handle = server.scene.add_mesh_trimesh(f"/h1/{link.attrib['name']}/{index}", geometry)
                visuals.append((demo.robot.model.getFrameId(link.attrib["name"]), handle, xyz, Rotation.from_euler("xyz", rpy).as_matrix()))
        com_marker = server.scene.add_icosphere("/com", radius=.025, color=(34, 197, 94))
        target_marker = server.scene.add_icosphere("/target", radius=.015, color=(245, 158, 11))
        projected_com = server.scene.add_icosphere("/com_projection", radius=.02, color=(34, 197, 94))
        projected_target = server.scene.add_icosphere("/target_projection", radius=.012, color=(245, 158, 11))
        overlays = server.gui.add_checkbox("显示支撑范围与受力", initial_value=True)
        server.gui.add_markdown("绿色：实际质心及地面投影；橙色：目标。\n\n"
                                "蓝框：假设足底接触范围；绿色竖线：支撑力（500 N/m）。")
        contact_visuals = []
        for index, anchor in enumerate(demo.controller.anchors):
            corners = np.array([anchor.translation + anchor.rotation @ [x, y, .006]
                                for x, y in ((-.10, -.035), (.10, -.035), (.10, .035), (-.10, .035))])
            patch = server.scene.add_line_segments(f"/support/{index}/patch",
                      np.stack((corners, np.roll(corners, -1, axis=0)), axis=1), colors=(59, 130, 246), thickness=.004)
            force = server.scene.add_line_segments(f"/support/{index}/force",
                      np.array([[anchor.translation, anchor.translation + [0., 0., .001]]]),
                      colors=(34, 197, 94), thickness=.008)
            label = server.scene.add_label(f"/support/{index}/label", "", position=anchor.translation + [0., 0., .65])
            contact_visuals.append((patch, force, label))
        force_values = np.zeros(len(contact_visuals))
        preview_marker = preview_path = support_outline = None
        if controller == "mpc":
            vertices = np.column_stack((demo.planner.support_vertices, np.full(len(demo.planner.support_vertices), .008)))
            support_outline = server.scene.add_line_segments("/support/hull",
                np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1), colors=(125, 211, 252), thickness=.003)
            preview_marker = server.scene.add_icosphere("/planned_zmp", radius=.012, color=(217, 70, 239), visible=False)
            preview_path = server.scene.add_line_segments("/zmp_preview", np.zeros((1, 2, 3)),
                colors=(217, 70, 239), thickness=.003, visible=False)
            server.gui.add_markdown("紫色：MPC 规划 ZMP 与未来 2 秒预览；浅蓝：双脚支撑多边形。")

        @server.on_client_connect
        def camera(client):
            client.camera.position = (1.9, -2.2, 1.5)
            client.camera.look_at = (0., 0., .9)

        def render():
            r = demo.robot
            terms = r.terms(demo.state)
            with server.atomic():
                for frame, handle, xyz, rotation in visuals:
                    pose = r.data.oMf[frame]
                    handle.position = pose.translation + pose.rotation @ xyz
                    handle.wxyz = Rotation.from_matrix(pose.rotation @ rotation).as_quat()[[3, 0, 1, 2]]
                com_marker.position = terms[5]
                target_marker.position = demo.target(demo.state.stamp)
                projected_com.position = [*terms[5][:2], .018]
                projected_target.position = [*demo.target(demo.state.stamp)[:2], .018]
                projected_com.visible = projected_target.visible = overlays.value
                if preview_marker is not None:
                    plan = demo.planner.last_plan
                    preview_marker.visible = preview_path.visible = overlays.value and plan is not None
                    support_outline.visible = overlays.value
                    if plan is not None:
                        p, _, a = plan.sample(demo.state.stamp)
                        preview_marker.position = [*(p - plan.height / plan.gravity * a), .024]
                        zmp = np.column_stack((plan.zmp, np.full(len(plan.zmp), .012)))
                        preview_path.points = np.stack((zmp[:-1], zmp[1:]), axis=1)
                for index, (patch, force, label) in enumerate(contact_visuals):
                    patch.visible = force.visible = label.visible = overlays.value
                    base = demo.controller.anchors[index].translation
                    force.points = np.array([[base, base + [0., 0., max(0., force_values[index]) / 500.]]])
                    label.position = base + [0., 0., max(0., force_values[index]) / 500. + .06]
                    label.text = f"{'左' if index == 0 else '右'}脚 {force_values[index]:.0f} N"

        render()
        print(f"H1 dynamics ready: {len(visuals)} meshes; autoplay={autoplay}, loop={loop}", flush=True)
        while True:
            start = time.monotonic()
            if reset_requested.is_set():
                reset_requested.clear()
                step_requested.clear()
                push_requested.clear()
                running.value = False
                demo.reset()
                force_values[:] = 0.
                render()
                status.content = "已复位并暂停"
            single = step_requested.is_set()
            step_requested.clear()
            if single:
                running.value = False
            if running.value or single:
                if push_requested.is_set():
                    push_requested.clear()
                    demo.trigger_push()
                previous_cycle = int((demo.state.stamp + 1e-9) / demo.duration)
                try:
                    for _ in range(1 if single else 4):
                        row = demo.step()
                        if not looping.value and int((demo.state.stamp + 1e-9) / demo.duration) > previous_cycle:
                            running.value = False
                            break
                    force_values[:] = [row["left_fz_n"], row["right_fz_n"]][:len(force_values)]
                    render()
                    status.content = (f"第 {int(row['time'] / demo.duration) + 1} 轮 · {row['time'] % demo.duration:.3f} / 8 s · 左/右支撑力 "
                                      f"{row['left_fz_n']:.1f} / {row['right_fz_n']:.1f} N\n\n"
                                      f"最大力矩 {row['max_torque_nm']:.1f} Nm · 接触位置误差 "
                                      f"{row['contact_position_error_m'] * 1000:.3f} mm\n\n"
                                      f"质心跟踪误差 {row['com_error_m'] * 1000:.1f} mm · WBC {row['wbc_ms']:.2f} ms\n\n"
                                      f"当前侧向外力 {row['applied_push_y_n']:.1f} N")
                    if row["solver_recoveries"]:
                        status.content += f"\n\n数值求解重建 {row['solver_recoveries']} 次（结果均经约束检查）"
                    if controller == "mpc":
                        status.content += f"\n\nMPC 重规划 {row['mpc_replans']} 次 · 最近规划 {row['last_mpc_ms']:.2f} ms"
                        if row["mpc_continuity_fallbacks"]:
                            status.content += f"\n\nMPC 加速度连续性回退 {row['mpc_continuity_fallbacks']} 次"
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    running.value = False
                    status.content = f"已暂停：{exc}"
                    print(status.content, flush=True)
            time.sleep(max(.001, .02 / speed.value - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--push-force", type=float, default=0., help="World-Y pelvis push, 0–5 N, at 2 s for 0.2 s each cycle")
    parser.add_argument("--controller", choices=("wbc", "mpc"), default="wbc")
    parser.add_argument("--motion", choices=("stand", "shift"), default="shift")
    parser.add_argument("--amplitude", type=float, default=.035, help="lateral CoM target amplitude in metres, 0–0.05")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--autoplay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.viewer:
        if args.csv is not None or args.steps != 1600:
            parser.error("--csv and --steps are headless-only")
        view(args.urdf, args.motion, args.amplitude, args.host, args.port, args.autoplay, args.loop, args.controller, args.push_force)
    else:
        print(json.dumps(run(args.urdf, args.motion, args.amplitude, args.steps, args.csv, args.controller, args.push_force), indent=2))
