"""Real Chromium/Viser smoke test for mixed WBC/MPC backends.

Requires optional viz dependencies, websockets, Chrome and local socket access.
Starts isolated processes and closes only those processes on completion.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from websockets.sync.client import connect

def check(backend="native", mpc_backend="native-sparse", output=Path("/tmp/control-browser-check")):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import wholebodyx
    package_root = Path(wholebodyx.__file__).parent
    files = list(package_root.rglob("*.py")) + [Path(__file__).resolve()]
    if backend == "native" or mpc_backend.startswith("native"):
        import wholebodyx_native
        files.append(Path(wholebodyx_native.__file__).resolve())
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="control-browser-") as directory:
        directory = Path(directory)
        log = (output / "server.log").open("w")
        app = subprocess.Popen([sys.executable, "-m", "wholebodyx.visualization",
                                "--backend", backend, "--mpc-backend", mpc_backend,
                                "--scenario", "disturbance_recovery", "--port", str(port)],
                               cwd="/tmp", stdout=log, stderr=subprocess.STDOUT)
        chrome = None
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if app.poll() is not None:
                    raise RuntimeError((output / "server.log").read_text())
                try:
                    with urllib.request.urlopen(url, timeout=.5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.1)
            else:
                raise TimeoutError("Viser failed to start")
            chrome = subprocess.Popen(["google-chrome", "--headless", "--no-sandbox", "--disable-dev-shm-usage",
                                       "--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--remote-debugging-port=0",
                                       f"--user-data-dir={directory / 'chrome'}", "about:blank"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            port_file = directory / "chrome/DevToolsActivePort"
            for _ in range(100):
                if port_file.exists():
                    break
                time.sleep(.1)
            debugger_port = port_file.read_text().splitlines()[0]
            with urllib.request.urlopen(f"http://127.0.0.1:{debugger_port}/json/list", timeout=3) as response:
                tabs = json.load(response)
            page = next(tab for tab in tabs if tab.get("type") == "page")
            errors = []
            with connect(page["webSocketDebuggerUrl"], proxy=None, max_size=20000000) as ws:
                seq = 0

                def call(method, params=None):
                    nonlocal seq
                    seq += 1
                    ws.send(json.dumps(dict(id=seq, method=method, params=params or {})))
                    while True:
                        reply = json.loads(ws.recv(timeout=15))
                        if reply.get("method") == "Runtime.exceptionThrown":
                            errors.append(reply)
                        if reply.get("id") == seq:
                            if "error" in reply:
                                raise RuntimeError(reply)
                            return reply.get("result", {})

                def evaluate(expression):
                    result = call("Runtime.evaluate", dict(expression=expression, returnByValue=True))
                    if "exceptionDetails" in result:
                        raise RuntimeError(result)
                    return result["result"].get("value")

                def body():
                    return evaluate("document.body?.innerText || ''")

                def wait(predicate, message, seconds=30):
                    deadline = time.monotonic() + seconds
                    while time.monotonic() < deadline:
                        text = body()
                        if "已暂停：" in text:
                            raise AssertionError(text)
                        if predicate(text):
                            return text
                        time.sleep(.1)
                    raise AssertionError((message, body()))

                def button(label):
                    evaluate("[...document.querySelectorAll('button')].find(e=>e.textContent.trim()===" + json.dumps(label) + ").click()")

                def checkbox(label):
                    evaluate("document.getElementById([...document.querySelectorAll('label')].find(e=>e.textContent.trim()===" + json.dumps(label) + ").htmlFor).click()")

                call("Runtime.enable")
                call("Emulation.setDeviceMetricsOverride", dict(width=1440, height=1050, deviceScaleFactor=1, mobile=False))
                call("Page.navigate", dict(url=url))
                wait(lambda text: "单步" in text and "MPC 后端" in text, "initial controls")
                button("单步")
                wait(lambda text: f"WBC {backend} / MPC {mpc_backend}" in text and "时间 0.02 s" in text, "mixed sparse step")
                checkbox("运行")
                wait(lambda text: "状态偏移 2 次" in text and "偏差重规划 2 次" in text, "two disturbance recoveries", 30)
                checkbox("运行")
                def stamp():
                    match = re.search(r"时间 ([\d.]+) s", body())
                    return None if match is None else float(match[1])
                # Confirm display convergence: queued WebSocket frames can
                # arrive after the server has already acknowledged a pause.
                pause_trace = []
                started = last_change = time.monotonic()
                paused = stamp()
                while time.monotonic() - started < 5:
                    time.sleep(.05)
                    current = stamp()
                    now = time.monotonic()
                    pause_trace.append(dict(elapsed_s=now - started, displayed_s=current))
                    if current != paused:
                        paused, last_change = current, now
                    if paused is not None and now - last_change >= 2:
                        break
                else:
                    raise AssertionError("pause display did not converge")
                time.sleep(.3)
                assert paused is not None and stamp() == paused, "pause failed"
                (output / "pause_trace.json").write_text(json.dumps(pause_trace, indent=2) + "\n")
                canvas_count = evaluate("document.querySelectorAll('canvas').length")
                assert canvas_count >= 3, canvas_count
                shot = call("Page.captureScreenshot", dict(format="png"))
                (output / "viewer.png").write_bytes(base64.b64decode(shot["data"]))
                button("重置机器人与曲线")
                wait(lambda text: "已重置" in text and "时间 0.00 s" in text, "reset")
                button("单步")
                wait(lambda text: f"WBC {backend} / MPC {mpc_backend}" in text and "时间 0.02 s" in text, "step after reset")
                assert not errors, errors
                report = dict(status="passed", wbc=backend, mpc=mpc_backend,
                              checks=["load", "mixed_backend_step", "two_disturbance_recoveries",
                                      "pause", "render_canvases", "reset", "step_after_reset"],
                              canvas_count=canvas_count, paused_at_s=paused, browser_exceptions=errors,
                              source_sha256=hashes,
                              source_unchanged=all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
                                                   for name, digest in hashes.items()))
                (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report), flush=True)
        finally:
            for process in (chrome, app):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "native"), default="native")
    parser.add_argument("--mpc-backend", choices=("python", "native", "sparse", "native-sparse"), default="native-sparse")
    parser.add_argument("--output", type=Path, default=Path("/tmp/control-browser-check"))
    args = parser.parse_args()
    check(args.backend, args.mpc_backend, args.output)
