"""Unitree H1 arm, bow, pickup and marching kinematic MPC/WBC demonstrations."""

import argparse
import csv
import json
from pathlib import Path
import time
import threading
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from wholebodyx import BackendRegistry, JointLimits, MPCReferenceManager, RobotState
from wholebodyx.model import PinocchioModel
from wholebodyx.simulation import JointIntegrator
from wholebodyx.tasks import FramePoseTask, PostureTask

DEFAULT_URDF = Path(__file__).resolve().parents[3] / "HumanoidAssets/UnitreeH1/robot.urdf"
ARMS = ("left_elbow_link", "right_elbow_link")
FEET = ("left_ankle_link", "right_ankle_link")


class H1Session:
    dt = .02

    def __init__(self, urdf=DEFAULT_URDF, mode="mpc", backend="python", mpc_backend=None, motion="arms", amplitude=1.25):
        if mode not in ("mpc", "wbc"):
            raise ValueError("mode must be mpc or wbc")
        self.urdf, self.mode = Path(urdf).resolve(), mode
        if motion not in ("arms", "bow", "pickup", "march"):
            raise ValueError("motion must be arms, bow, pickup or march")
        if not np.isfinite(amplitude) or not .5 <= amplitude <= 1.4:
            raise ValueError("amplitude must be between 0.5 and 1.4")
        self.amplitude = float(amplitude)
        self.motion = motion
        self.model = PinocchioModel(self.urdf)
        names = self.model.joint_names
        if len(names) != 19 or "torso_joint" not in names:
            raise ValueError("expected the 19-joint Unitree H1 model")
        self.indices = {name: i for i, name in enumerate(names)}
        bounds = self.model.limits
        # Demo velocity cap and slew limit, not manufacturer acceleration data.
        velocity_cap, acceleration_cap = (4., 24.) if motion == "march" else (.6, 2.)
        self.limits = JointLimits(bounds.lower, bounds.upper, np.minimum(bounds.velocity, velocity_cap), np.full(19, acceleration_cap))
        registry = BackendRegistry()
        self.controller = registry.create_controller(backend, self.limits)
        self.manager = (MPCReferenceManager(registry.create_planner(mpc_backend or backend, self.limits), self.limits)
                        if mode == "mpc" else None)
        self.reset()

    def reset(self):
        self.plant = JointIntegrator(self.limits, RobotState(np.zeros(19), np.zeros(19)))
        self.controller.reset()
        if self.manager is not None:
            self.manager.reset()
        self.feet = [self.model.pose(self.plant.read().q, frame) for frame in FEET]
        self.steps = self.replans = 0
        self.max_violation = self.max_foot_error = 0.
        self.max_pelvis_pitch = 0.
        self.max_foot_lift = 0.
        self.max_foot_separation = 0.
        self.object_attached = False
        self.grasp_time = None
        self.object_position = self.hand_transform(self.goal(3.))[1].copy()
        self.object_initial = self.object_position.copy()
        self.object_rotation = np.eye(3)
        self._grasp_offset = np.zeros(3)
        self._grasp_rotation = np.eye(3)

    def hand_transform(self, q):
        """Virtual grasp point at the distal end of the supplied forearm mesh."""
        world_r, world_t = self.world_transform(q)
        hand = self.model.pose(q, "right_elbow_link")
        return (world_r @ hand.rotation,
                world_r @ (hand.translation + hand.rotation @ np.array([.27, 0, 0])) + world_t)

    def world_transform(self, q):
        """Kinematic support-foot anchoring, not a floating-base dynamics solve."""
        if self.motion == "arms":
            return np.eye(3), np.zeros(3)
        if self.motion == "march":
            # Display-only root: keep the lower ankle at its initial height.
            # This is geometric floor alignment, not a contact/balance solve.
            feet = [self.model.pose(q, frame).translation for frame in FEET]
            shift = np.array([0., -.5 * (feet[0][1] + feet[1][1]),
                              min(p.translation[2] for p in self.feet) - min(p[2] for p in feet)])
            return np.eye(3), shift
        foot = self.model.pose(q, FEET[0])
        rotation = self.feet[0].rotation @ foot.rotation.T
        return rotation, self.feet[0].translation - rotation @ foot.translation

    def march_target(self, stamp, velocity=False):
        # Three full gait cycles in six seconds, with smooth entry/exit.
        t = stamp % 8.
        u = np.clip(t / .6, 0., 1.) * np.clip((6.6 - t) / .6, 0., 1.)
        envelope = u * u * (3 - 2 * u)
        du = 1 / .6 if 0 < t < .6 else (-1 / .6 if 6. < t < 6.6 else 0.)
        envelope_rate = 6 * u * (1 - u) * du
        wave, wave_rate = np.sin(np.pi * t), np.pi * np.cos(np.pi * t)
        q = np.zeros(19)
        for side, sign in (("left", 1), ("right", -1)):
            swing = sign * wave
            knee = .16 + .8 * max(swing, 0.) ** 2
            stride = -.32 * swing
            values = (("hip_pitch", stride - .5 * knee), ("knee", knee),
                      ("ankle", -stride - .5 * knee), ("hip_roll", .045 * wave),
                      ("shoulder_pitch", .4 * swing), ("shoulder_roll", sign * .10),
                      ("elbow", .28 + .10 * swing))
            swing_rate = sign * wave_rate
            knee_rate = 1.6 * max(swing, 0.) * swing_rate
            stride_rate = -.32 * swing_rate
            rates = (stride_rate - .5 * knee_rate, knee_rate, -stride_rate - .5 * knee_rate,
                     .045 * wave_rate, .4 * swing_rate, 0., .10 * swing_rate)
            for (joint, value), rate in zip(values, rates):
                q[self.indices[f"{side}_{joint}_joint"]] = self.amplitude * (
                    rate * envelope + value * envelope_rate if velocity else value * envelope)
        q[self.indices["torso_joint"]] = -.10 * self.amplitude * (
            wave_rate * envelope + wave * envelope_rate if velocity else wave * envelope)
        return q

    def goal(self, stamp):
        q = np.zeros(19)
        if self.motion == "march":
            return self.march_target(stamp)
        if self.motion in ("bow", "pickup"):
            # Cubic easing gives zero target velocity at the start/end of
            # each bend. The solver still enforces velocity and slew bounds.
            progress = np.clip((stamp - .5) / 2., 0., 1.) if stamp < 4. else 1 - np.clip((stamp - 4.) / 2., 0., 1.)
            amount = progress * progress * (3 - 2 * progress)
            if amount > 0:
                for side in ("left", "right"):
                    for joint, value in (("hip_pitch", -.8), ("knee", .35), ("ankle", -.15),
                                         ("shoulder_pitch", -.15), ("elbow", .3)):
                        q[self.indices[f"{side}_{joint}_joint"]] = value * amount * self.amplitude
            return q
        # Return to a reachable neutral posture after two modest arm poses.
        if stamp < 4. - 1e-9:
            second = stamp >= 2. - 1e-9
            for side, sign in (("left", 1), ("right", -1)):
                for joint, value in (("shoulder_pitch", -.3 if second else -.15),
                                     ("shoulder_roll", sign * (.2 if second else .1)),
                                     ("elbow", .6 if second else .35)):
                    q[self.indices[f"{side}_{joint}_joint"]] = value
        return q

    def phase(self):
        stamp = self.plant.read().stamp
        if self.motion == "march":
            return "收步回正" if stamp % 8 >= 6 else "全身交替迈步 · 运动学"
        if self.motion == "arms":
            return "上肢运动" if stamp < 4 else "回到初始姿态"
        if stamp < .5:
            return "准备"
        if stamp < 2.5:
            return "缓慢前俯"
        if stamp < 3:
            return "对准物体" if self.motion == "pickup" else "保持前俯"
        if stamp < 4:
            return ("已抓取 · 准备抬起" if self.object_attached else "等待接近物体") if self.motion == "pickup" else "保持前俯"
        if stamp < 6:
            return "回正并抬起" if self.object_attached else "缓慢回正"
        return "完成 · 持物展示" if self.object_attached else "已回正"

    def goal_velocity(self, stamp):
        if self.motion == "march":
            return self.march_target(stamp, velocity=True)
        rate = 0.
        if .5 < stamp < 2.5:
            u = (stamp - .5) / 2.
            rate = 3 * u * (1 - u)
        elif 4. < stamp < 6.:
            u = (stamp - 4.) / 2.
            rate = -3 * u * (1 - u)
        return self.goal(3.) * rate if self.motion != "arms" else np.zeros(19)

    def step(self):
        state = self.plant.read()
        goal = self.goal(state.stamp)
        qref, vref, reason = goal, self.goal_velocity(state.stamp), "disabled"
        if self.manager is not None:
            update = self.manager.update(state, goal, self.dt)
            if not update.success:
                raise RuntimeError(f"MPC failed: {update.solver}")
            qref, vref, reason = update.position, update.velocity, update.reason
            self.replans += int(update.replanned)
        tasks = [PostureTask(qref, feedforward=vref).linearize(self.model, state)]
        for frame in ARMS:
            target = self.model.pose(qref, frame)
            twist = self.model.jacobian(qref, frame) @ vref
            tasks.append(FramePoseTask(frame, target, feedforward=twist).linearize(self.model, state))
        for frame, target in zip(FEET, self.feet):
            if self.motion in ("bow", "pickup", "march"):
                target = self.model.pose(qref, frame)
                twist = self.model.jacobian(qref, frame) @ vref
            else:
                twist = None
            tasks.append(FramePoseTask(frame, target, weight=10., feedforward=twist).linearize(self.model, state))
        start = time.perf_counter()
        result = self.controller.step(state, tasks, self.dt)
        elapsed = (time.perf_counter() - start) * 1000
        if not result.solver.success or result.command is None:
            raise RuntimeError(f"WBC failed: {result.solver}")
        self.plant.write(result.command, self.dt)
        self.steps += 1
        state = self.plant.read()
        hand_r, hand_p = self.hand_transform(state.q)
        grasp_distance = float(np.linalg.norm(hand_p - self.object_position))
        if self.motion == "pickup":
            if not self.object_attached and 3. <= state.stamp < 4. and grasp_distance <= .015:
                self.object_attached = True
                self.grasp_time = state.stamp
                self._grasp_offset = hand_r.T @ (self.object_position - hand_p)
                self._grasp_rotation = hand_r.T @ self.object_rotation
            if self.object_attached:
                self.object_position = hand_p + hand_r @ self._grasp_offset
                self.object_rotation = hand_r @ self._grasp_rotation
        world_r, world_t = self.world_transform(state.q)
        pitch = float(Rotation.from_matrix(world_r).as_euler("xyz")[1])
        self.max_pelvis_pitch = max(self.max_pelvis_pitch, abs(pitch))
        foot_targets = self.feet
        if self.motion == "march":
            target_q = self.goal(state.stamp)
            target_r, target_t = self.world_transform(target_q)
            foot_targets = [self.model.pose(target_q, frame) for frame in FEET]
            target_positions = [target_r @ target.translation + target_t for target in foot_targets]
        else:
            target_positions = [target.translation for target in foot_targets]
        lifts = [float((world_r @ self.model.pose(state.q, frame).translation + world_t - initial.translation)[2])
                 for frame, initial in zip(FEET, self.feet)]
        self.max_foot_lift = max(self.max_foot_lift, *lifts)
        foot_error = max(np.linalg.norm(world_r @ self.model.pose(state.q, frame).translation + world_t - target)
                         for frame, target in zip(FEET, target_positions))
        self.max_foot_error = max(self.max_foot_error, float(foot_error))
        self.max_violation = max(self.max_violation, result.solver.violation)
        row = dict(time=state.stamp, joint_error_rad=float(np.max(np.abs(state.q - self.goal(state.stamp)))),
                   foot_error_m=float(foot_error), wbc_ms=elapsed, replan_reason=reason,
                   violation=result.solver.violation)
        foot_x = [float((world_r @ self.model.pose(state.q, frame).translation + world_t)[0]) for frame in FEET]
        self.max_foot_separation = max(self.max_foot_separation, abs(foot_x[0] - foot_x[1]))
        row["left_foot_x_m"], row["right_foot_x_m"] = foot_x
        row["pelvis_y_m"], row["pelvis_z_m"] = float(world_t[1]), float(world_t[2])
        row["left_foot_lift_m"], row["right_foot_lift_m"] = lifts
        row["pelvis_pitch_rad"] = pitch
        row["object_attached"] = self.object_attached
        row["grasp_distance_m"] = grasp_distance
        row["object_lift_m"] = float(self.object_position[2] - self.object_initial[2])
        for axis, value in zip("xyz", self.object_position):
            row[f"object_{axis}_m"] = float(value)
        for i, name in enumerate(self.model.joint_names):
            row[f"q_{name}"] = float(state.q[i])
            row[f"v_{name}"] = float(state.v[i])
        return row


def run(urdf=DEFAULT_URDF, mode="mpc", backend="python", mpc_backend=None, steps=400, csv_path=None, motion="arms", amplitude=1.25):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    session = H1Session(urdf, mode, backend, mpc_backend, motion, amplitude)
    stream = None
    try:
        if csv_path is not None:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("w", newline="")
        writer = None
        for _ in range(steps):
            row = session.step()
            if stream is not None:
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
    finally:
        if stream is not None:
            stream.close()
    return dict(urdf=str(session.urdf), mode=mode, motion=motion, amplitude=amplitude, fixed_pelvis=motion == "arms", joints=19, steps=steps,
                solver_fixed_base=True,
                base_model="floor_aligned_kinematic" if motion == "march" else ("support_foot_kinematic" if motion in ("bow", "pickup") else "fixed"),
                max_pelvis_pitch_rad=session.max_pelvis_pitch, max_foot_lift_m=session.max_foot_lift, max_fore_aft_foot_separation_m=session.max_foot_separation,
                object_attached=session.object_attached, grasp_time_s=session.grasp_time,
                object_lift_m=row["object_lift_m"],
                duration_s=row["time"], final_joint_error_rad=row["joint_error_rad"],
                max_foot_error_m=session.max_foot_error, max_constraint_violation=session.max_violation,
                replans=session.replans, backend=backend, mpc_backend=mpc_backend or backend)


def view(urdf, mode, backend, mpc_backend, host, port, motion="arms", autoplay=True, loop=True, amplitude=1.25):
    import trimesh
    import viser

    session = H1Session(urdf, mode, backend, mpc_backend, motion, amplitude)
    server = viser.ViserServer(host=host, port=port)
    try:
        server.scene.set_up_direction("+z")
        server.scene.add_grid("/ground", width=4, height=4)
        description = ("弯腰拾取：前俯接近台上物体，3 秒后接近手端时附着，4 秒起回正持物。"
                       "\n\n虚拟手端抓取，不模拟手指与物体接触。" if motion == "pickup" else
                       "脚部锚定的运动学弯腰展示：0.5 秒开始前俯，保持后于 4 秒回正。"
                       if motion == "bow" else "固定骨盆：0–2 s 抬臂，2–4 s 换目标，4 s 后回到零位。")
        if motion == "march":
            description = "全身迈步：每秒交替迈一步，配合屈膝、踝补偿、骨盆起伏、躯干转动和对侧摆臂。每轮含收步回正；运动学展示，无动力学或前进位移。"
        server.gui.add_markdown(f"# H1 MPC + WBC\n{description}\n\n不含平衡、接触力或碰撞控制。")
        running = server.gui.add_checkbox("运行", initial_value=autoplay)
        looping = server.gui.add_checkbox("循环播放", initial_value=loop)
        speed = server.gui.add_slider("播放速度", min=.25, max=1.5, step=.25, initial_value=1.)
        reset_requested, step_requested = threading.Event(), threading.Event()
        server.gui.add_button("重新开始").on_click(lambda _: reset_requested.set())
        server.gui.add_button("单步观察").on_click(lambda _: step_requested.set())
        status = server.gui.add_markdown("默认自动运行并循环；取消“运行”可暂停，关闭“循环播放”将在本轮结束后停止。")
        visuals = []
        for link in ET.parse(session.urdf).getroot().findall("link"):
            for i, visual in enumerate(link.findall("visual")):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                path = session.urdf.parent / mesh.attrib["filename"]
                geometry = trimesh.load(path, force="mesh")
                geometry.apply_scale(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "))
                origin = visual.find("origin")
                xyz = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                handle = server.scene.add_mesh_trimesh(f"/h1/{link.attrib['name']}/{i}", geometry)
                visuals.append((link.attrib["name"], handle, xyz, Rotation.from_euler("xyz", rpy).as_matrix()))
        q0 = session.plant.read().q
        # This H1 ankle mesh extends 0.07 m below its link origin.
        height = -min(session.model.pose(q0, f).translation[2] for f in FEET) + .071
        prop = None
        grasp_marker = None
        if motion == "pickup":
            center = session.object_initial + [0, 0, height]
            prop = server.scene.add_box("/object", dimensions=(.06, .06, .06), color=(245, 158, 11), position=center)
            grasp_marker = server.scene.add_icosphere("/grasp_point", radius=.009, color=(56, 189, 248))
            top = center[2] - .03
            server.scene.add_box("/table", dimensions=(.28, .28, .04), color=(100, 116, 139),
                                 position=(center[0], center[1], top - .02))
            server.scene.add_box("/table_leg", dimensions=(.06, .06, top - .04), color=(71, 85, 105),
                                 position=(center[0], center[1], (top - .04) / 2))

        @server.on_client_connect
        def camera(client):
            client.camera.position = (2.3, -2.6, 1.65)
            client.camera.look_at = (.15, -.08, .95)

        def render():
            q = session.plant.read().q
            world_r, world_t = session.world_transform(q)
            with server.atomic():
                for frame, handle, xyz, rotation in visuals:
                    pose = session.model.pose(q, frame)
                    handle.position = world_r @ (pose.translation + pose.rotation @ xyz) + world_t + [0, 0, height]
                    handle.wxyz = np.roll(Rotation.from_matrix(world_r @ pose.rotation @ rotation).as_quat(), 1)
                if prop is not None:
                    prop.position = session.object_position + [0, 0, height]
                    prop.wxyz = np.roll(Rotation.from_matrix(session.object_rotation).as_quat(), 1)
                    prop.color = (52, 211, 153) if session.object_attached else (245, 158, 11)
                    grasp_marker.position = session.hand_transform(q)[1] + [0, 0, height]
                    grasp_marker.visible = not session.object_attached

        render()
        print(f"H1 ready: {len(visuals)} visual meshes, 19 joints; autoplay={autoplay}, loop={loop}.", flush=True)
        while True:
            started = time.monotonic()
            if reset_requested.is_set():
                reset_requested.clear()
                step_requested.clear()
                running.value = False
                session.reset()
                render()
                status.content = "已复位 · 点击运行或单步观察。"
            single_step = step_requested.is_set()
            step_requested.clear()
            if single_step:
                running.value = False
            if running.value or single_step:
                if session.steps >= 400:
                    session.reset()
                try:
                    row = session.step()
                    render()
                    status.content = (f"**{session.phase()}** · {row['time']:.2f} / 8.00 s\n\n"
                                      f"前俯 {np.degrees(row['pelvis_pitch_rad']):.1f}° · "
                                      f"WBC {row['wbc_ms']:.2f} ms · MPC 重规划 {session.replans} 次")
                    if motion == "march":
                        status.content += f"\n\n左脚抬高 {row['left_foot_lift_m'] * 100:.1f} cm · 右脚 {row['right_foot_lift_m'] * 100:.1f} cm"
                    if motion == "pickup":
                        status.content += (f"\n\n{'已拾取' if session.object_attached else '接近物体'}"
                                           f" · 抬升 {row['object_lift_m']:.3f} m")
                    if session.steps >= 400 and not looping.value:
                        running.value = False
                except (ValueError, RuntimeError) as exc:
                    running.value = False
                    status.content = f"已暂停：{exc}"
            time.sleep(max(.001, session.dt / speed.value - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mode", choices=("mpc", "wbc"), default="mpc")
    parser.add_argument("--motion", choices=("arms", "bow", "pickup", "march"), default="arms")
    parser.add_argument("--amplitude", type=float, default=1.25, help="Bow/pickup/march scale, 0.5–1.4 (default: 1.25)")
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    parser.add_argument("--mpc-backend", choices=("python", "native"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--autoplay", action=argparse.BooleanOptionalAction, default=True,
                        help="Start the viewer immediately (default: enabled)")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True,
                        help="Repeat the viewer animation (default: enabled)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    if args.viewer:
        if args.csv is not None or args.steps != 400:
            parser.error("--csv and --steps are headless-only")
        view(args.urdf, args.mode, args.backend, args.mpc_backend, args.host, args.port, args.motion,
             args.autoplay, args.loop, args.amplitude)
    else:
        print(json.dumps(run(args.urdf, args.mode, args.backend, args.mpc_backend, args.steps, args.csv, args.motion, args.amplitude), indent=2))
