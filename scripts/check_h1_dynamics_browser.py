"""Real Chromium/Viser smoke test for the H1 dynamics viewer.

Requires optional viz dependencies, websockets, Chrome and local socket access.
Starts isolated processes and closes only those processes on completion.
"""
import argparse
import base64
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

ROOT = Path(__file__).resolve().parents[1]


def check(controller="mpc", output=Path("/tmp/h1-browser-check")):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="h1-browser-") as directory:
        directory = Path(directory)
        log = (output / "server.log").open("w")
        app = subprocess.Popen([sys.executable, str(ROOT / "examples/python/h1_dynamics_wbc.py"),
                                "--viewer", "--controller", controller, "--port", str(port), "--no-autoplay"],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
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

                def time_in(text):
                    match = re.search(r"第 (\d+) 轮 · ([\d.]+) / 8", text)
                    return None if match is None else (int(match[1]) - 1) * 8 + float(match[2])

                call("Runtime.enable")
                call("Emulation.setDeviceMetricsOverride", dict(width=1440, height=1050, deviceScaleFactor=1, mobile=False))
                call("Page.navigate", dict(url=url))
                wait(lambda text: "单步 5 ms" in text and "正在初始化" in text, "initial controls")
                button("单步 5 ms")
                wait(lambda text: time_in(text) == .005, "single step")
                checkbox("运行")
                wait(lambda text: time_in(text) is not None and time_in(text) > 10.0, "continuous second cycle", 40)
                checkbox("运行")
                time.sleep(.3)
                paused = time_in(body())
                time.sleep(.3)
                assert time_in(body()) == paused, "pause failed"
                shot = call("Page.captureScreenshot", dict(format="png"))
                (output / "viewer.png").write_bytes(base64.b64decode(shot["data"]))
                checkbox("循环播放")
                checkbox("运行")
                wait(lambda text: time_in(text) is not None and time_in(text) >= 16., "stop at cycle boundary", 40)
                time.sleep(.3)
                assert time_in(body()) == 16., body()
                # Exercise a queued push while paused, then resume physics.
                button("侧向轻推 3 N · 0.2 s")
                # Freeze the first disturbed step so slow CDP polling cannot
                # miss the entire 0.2 s pulse while other checks use the CPU.
                button("单步 5 ms")
                wait(lambda text: "当前侧向外力 3.0 N" in text, "plant-only side push")
                assert time_in(body()) == 16.005, body()
                checkbox("运行")
                wait(lambda text: "当前侧向外力 0.0 N" in text, "push duration elapsed")
                checkbox("运行")
                button("复位并暂停")
                wait(lambda text: "已复位并暂停" in text, "reset")
                button("单步 5 ms")
                wait(lambda text: time_in(text) == .005, "step after reset")
                assert not errors, errors
                report = dict(controller=controller, status="passed", checks=["load", "single_step", "continuous_loop",
                              "pause", "stop_at_boundary", "side_push", "reset", "step_after_reset"], browser_exceptions=errors)
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
    parser.add_argument("--controller", choices=("wbc", "mpc"), default="mpc")
    parser.add_argument("--output", type=Path, default=Path("/tmp/h1-browser-check"))
    args = parser.parse_args()
    check(args.controller, args.output)
