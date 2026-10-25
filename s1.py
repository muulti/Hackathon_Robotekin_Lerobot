#!/usr/bin/env python3
"""Small macOS GUI to start/stop a LeRobot policy rollout.

Can also be imported: policy_server.py uses build_command() and
run_rollout_blocking() to run a policy without opening the GUI.
"""

import os
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
from pathlib import Path

#HERE ARE SOME PLACHOLDERS!!
POLICY_PATH = (
    "path_placeholder"
)
ROBOT_ID = "Desinformatico"
ROBOT_TYPE = "so101_follower"
ROBOT_PORT = "/dev/tty.usbmodem5B610343321"
DURATION_SECONDS = 15

CAMERAS = (
    "{Camara_Alta: {type: opencv, index_or_path: 0, width: 640, "
    "height: 480, fps: 30}, camara_pinza: {type: opencv, "
    "index_or_path: 1, width: 640, height: 480, fps: 30}}"
)


def make_cameras(top_name, gripper_name):
    """Same two physical cameras, but with the names a policy was trained on."""
    return (
        "{" + top_name + ": {type: opencv, index_or_path: 0, width: 640, "
        "height: 480, fps: 30}, " + gripper_name + ": {type: opencv, "
        "index_or_path: 1, width: 640, height: 480, fps: 30}}"
    )


def setPolicyPath(policy_path):
    """Change the default policy used by the GUI / build_command()."""
    global POLICY_PATH
    POLICY_PATH = policy_path


# Fallback for the Conda environment shown in your terminal logs.
ROLLOUT_FALLBACK = Path(
    "/Users/multi/miniforge3/envs/lerobot/bin/lerobot-rollout"
)


def find_rollout_command():
    found = shutil.which("lerobot-rollout")
    if found:
        return found
    if ROLLOUT_FALLBACK.is_file():
        return str(ROLLOUT_FALLBACK)
    raise FileNotFoundError(
        "Could not find lerobot-rollout. Activate your 'lerobot' "
        "environment and run this script from that terminal."
    )


def validate_setup(policy_path=None):
    """Return an error message if something is missing, else None."""
    policy_path = policy_path or POLICY_PATH
    if not Path(policy_path, "config.json").is_file():
        return f"Could not find config.json in: {policy_path}"
    if not Path(ROBOT_PORT).exists():
        return f"Could not find serial port: {ROBOT_PORT}"
    return None


def build_command(policy_path=None, cameras=None):
    """Build the lerobot-rollout command for the given policy."""
    policy_path = policy_path or POLICY_PATH
    cameras = cameras or CAMERAS
    return [
        find_rollout_command(),
        "--strategy.type=base",
        f"--policy.path={policy_path}",
        f"--robot.type={ROBOT_TYPE}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.cameras={cameras}",
        "--task=Pick up the object",
        f"--duration={DURATION_SECONDS}",
    ]


def run_rollout_blocking(policy_path=None, cameras=None, timeout=None):
    """Run a rollout without the GUI and wait until it finishes.

    Returns (return_code, combined_output).
    """
    command = build_command(policy_path, cameras)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        timeout=timeout or (DURATION_SECONDS + 120),
    )
    return result.returncode, result.stdout


def find_replay_command():
    found = shutil.which("lerobot-replay")
    if found:
        return found
    sibling = Path(find_rollout_command()).with_name("lerobot-replay")
    if sibling.is_file():
        return str(sibling)
    raise FileNotFoundError("Could not find lerobot-replay.")


def validate_replay(dataset_path):
    """Return an error message if the replay can't run, else None."""
    if not Path(dataset_path, "meta", "info.json").is_file():
        return f"Could not find a dataset (meta/info.json) in: {dataset_path}"
    if not Path(ROBOT_PORT).exists():
        return f"Could not find serial port: {ROBOT_PORT}"
    return None


def run_replay_blocking(dataset_path, episode=0, timeout=300):
    """Replay one episode of a local dataset on the robot and wait.

    Returns (return_code, combined_output).
    """
    command = [
        find_replay_command(),
        f"--robot.type={ROBOT_TYPE}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.port={ROBOT_PORT}",
        f"--dataset.repo_id=local/{Path(dataset_path).name}",
        f"--dataset.root={dataset_path}",
        f"--dataset.episode={episode}",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        timeout=timeout,
    )
    return result.returncode, result.stdout


class RolloutGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LeRobot Policy Runner")
        self.root.geometry("760x500")
        self.process = None
        self.reader_thread = None

        heading = tk.Label(
            root, text="SO-101 Policy Runner",
            font=("Helvetica", 17, "bold")
        )
        heading.pack(pady=(14, 6))

        self.status = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status).pack(pady=(0, 8))

        buttons = tk.Frame(root)
        buttons.pack(pady=5)

        self.start_button = tk.Button(
            buttons, text="Start rollout", command=self.start,
            width=16, height=2, bg="#2e7d32", fg="white"
        )
        self.start_button.pack(side=tk.LEFT, padx=8)

        self.stop_button = tk.Button(
            buttons, text="Stop", command=self.stop,
            width=12, height=2, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=8)

        tk.Label(
            root,
            text=f"Robot: {ROBOT_ID}   |   Duration: {DURATION_SECONDS}s",
        ).pack(pady=4)

        self.log = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, height=19, state=tk.DISABLED
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 12))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def append_log(self, text):
        def update():
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, text)
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)
        self.root.after(0, update)

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return

        if not Path(POLICY_PATH, "config.json").is_file():
            messagebox.showerror(
                "Policy not found",
                f"Could not find config.json in:\n{POLICY_PATH}"
            )
            return

        if not Path(ROBOT_PORT).exists():
            messagebox.showerror(
                "Robot port not found",
                f"Could not find serial port:\n{ROBOT_PORT}\n\n"
                "Check the port and reconnect the robot."
            )
            return

        confirmed = messagebox.askyesno(
            "Confirm physical robot movement",
            "This will connect to the SO-101 and run the trained policy "
            f"for up to {DURATION_SECONDS} seconds.\n\n"
            "Clear the workspace and keep your hand near the robot's "
            "power switch. Start now?"
        )
        if not confirmed:
            return

        try:
            command = build_command(POLICY_PATH)
        except FileNotFoundError as exc:
            messagebox.showerror("LeRobot not found", str(exc))
            return

        self.append_log("$ " + " ".join(command) + "\n\n")
        try:
            env = os.environ.copy()
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self.process = None
            messagebox.showerror("Could not start rollout", str(exc))
            return

        self.status.set("Running — keep the workspace clear")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.reader_thread = threading.Thread(
            target=self.read_output, daemon=True
        )
        self.reader_thread.start()

    def read_output(self):
        proc = self.process
        if proc is None:
            return
        try:
            if proc.stdout:
                for line in proc.stdout:
                    self.append_log(line)
            return_code = proc.wait()
        except Exception as exc:
            self.append_log(f"\nGUI output error: {exc}\n")
            return_code = -1

        def finished():
            self.status.set(
                "Finished successfully" if return_code == 0
                else f"Stopped/finished with exit code {return_code}"
            )
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.process = None
        self.root.after(0, finished)

    def stop(self):
        proc = self.process
        if proc is not None and proc.poll() is None:
            self.status.set("Stopping rollout…")
            self.append_log("\nStopping rollout…\n")
            proc.terminate()
            self.stop_button.configure(state=tk.DISABLED)

    def on_close(self):
        proc = self.process
        if proc is not None and proc.poll() is None:
            if not messagebox.askyesno(
                "Rollout is running",
                "Stop the rollout and close the window?"
            ):
                return
            proc.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RolloutGUI(root)
    root.mainloop()
