"""
J.A.R.V.I.S. - Personal Assistant Control Panel (single-file Streamlit app)
===========================================================================

INSTALL (run once):

    pip install "streamlit>=1.37" groq psutil
    pip install pypdf            # optional - lets JARVIS read uploaded PDFs

RUN:

    streamlit run app.py

NOTES:
  * Free Groq API key: https://console.groq.com/keys  (paste it in the sidebar,
    or export GROQ_API_KEY before launching).
  * Android control needs Google's platform-tools: `adb` must be on your PATH
    and USB debugging must be enabled on the phone.
  * File tools are sandboxed to ~/jarvis_workspace (override with the
    JARVIS_WORKSPACE environment variable).
  * Every tool that runs a command, launches software, modifies files or talks
    to the phone must be approved by you in the UI before it executes.
"""

from __future__ import annotations

import html
import inspect
import io
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

st.set_page_config(
    page_title="J.A.R.V.I.S.",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import psutil
except ImportError:
    st.error("Missing dependency. Run: `pip install psutil`")
    st.stop()

try:
    import groq as groq_sdk
    from groq import Groq
except ImportError:
    st.error("Missing dependency. Run: `pip install groq`")
    st.stop()


# ======================================================================
# Configuration
# ======================================================================
MODEL = "openai/gpt-oss-120b"
WORKSPACE = Path(
    os.environ.get("JARVIS_WORKSPACE", str(Path.home() / "jarvis_workspace"))
).expanduser().resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

SYSTEM_NAME = platform.system()  # "Windows" | "Darwin" | "Linux"
IS_WINDOWS = SYSTEM_NAME == "Windows"
IS_MAC = SYSTEM_NAME == "Darwin"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # 0 on non-Windows
DISK_ROOT = Path.home().anchor or "/"
GB = 1024 ** 3

MAX_TOOL_OUTPUT = 4000      # chars of tool output sent back to the model
MAX_ATTACH_CHARS = 8000     # chars of an uploaded document sent to the model
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_HISTORY = 24            # messages sent to the model per request
MAX_AGENT_STEPS = 6         # model calls per user turn (loop guard)

TEXT_EXTS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".log", ".py", ".js", ".ts", ".html",
    ".css", ".xml", ".yaml", ".yml", ".ini", ".toml", ".cfg", ".sql", ".sh",
    ".bat", ".ps1", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php", ".tex",
}

psutil.cpu_percent(interval=None)  # prime the non-blocking CPU counter

ss = st.session_state


# ======================================================================
# Small helpers
# ======================================================================
def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _as_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(lo, min(hi, number))


def _req_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be an integer")


def _fmt_uptime(seconds: float) -> str:
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    return (f"{days}d " if days else "") + f"{hours:02d}h {minutes:02d}m"


def _safe_path(rel: str = "") -> Path:
    """Resolve a user/model supplied path and keep it inside the workspace."""
    rel = (rel or "").strip().replace("\\", "/")
    candidate = (WORKSPACE / rel).resolve()
    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise ValueError("Path escapes the workspace sandbox.")
    return candidate


def _rel(path: Path) -> str:
    try:
        return path.relative_to(WORKSPACE).as_posix() or "."
    except ValueError:
        return str(path)


def get_api_key() -> str:
    return (ss.get("groq_api_key") or "").strip()


def get_title() -> str:
    return (ss.get("user_title") or "").strip() or "Sir"


def audit(event: str, tool: str, detail: str = "") -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = _clip(f"[{stamp}] {event:<9} {tool} {detail}".strip(), 300).replace("\n", " ")
    ss.audit_log.append(line)
    ss.audit_log[:] = ss.audit_log[-100:]
    try:
        with open(WORKSPACE / ".jarvis_audit.log", "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")
    except OSError:
        pass


# ======================================================================
# Tool handlers (each returns a JSON-serialisable dict)
# ======================================================================
# ---------- PC: system stats -------------------------------------------------
def tool_get_system_stats() -> dict:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    freq = None
    try:
        f = psutil.cpu_freq()
        freq = round(f.current) if f else None
    except Exception:
        pass

    disks = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        disks.append({
            "mount": part.mountpoint,
            "fs": part.fstype,
            "total_gb": round(usage.total / GB, 1),
            "used_gb": round(usage.used / GB, 1),
            "free_gb": round(usage.free / GB, 1),
            "percent": usage.percent,
        })
        if len(disks) >= 6:
            break

    battery = None
    try:
        b = psutil.sensors_battery()
        if b:
            battery = {"percent": round(b.percent), "plugged_in": b.power_plugged}
    except Exception:
        pass

    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_percent"]):
        try:
            info = p.info
            procs.append({
                "pid": info["pid"],
                "name": info["name"],
                "mem_percent": round(info["memory_percent"] or 0, 2),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procs.sort(key=lambda d: d["mem_percent"], reverse=True)

    return {
        "ok": True,
        "os": platform.platform(),
        "hostname": platform.node(),
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.4),
            "logical_cores": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "freq_mhz": freq,
        },
        "memory": {
            "percent": vm.percent,
            "used_gb": round(vm.used / GB, 2),
            "total_gb": round(vm.total / GB, 2),
            "swap_percent": swap.percent,
        },
        "disks": disks,
        "battery": battery,
        "uptime": _fmt_uptime(time.time() - psutil.boot_time()),
        "top_processes_by_memory": procs[:5],
    }


# ---------- PC: workspace files ---------------------------------------------
def tool_list_workspace_files(subdir: str = "", recursive: Any = False) -> dict:
    base = _safe_path(subdir)
    if not base.is_dir():
        raise ValueError(f"'{subdir}' is not a directory in the workspace.")
    iterator = base.rglob("*") if _as_bool(recursive) else base.iterdir()
    entries, truncated = [], False
    for p in sorted(iterator, key=lambda x: str(x).lower()):
        if any(part.startswith(".") for part in p.relative_to(base).parts):
            continue
        try:
            info = p.stat()
        except OSError:
            continue
        entries.append({
            "path": _rel(p),
            "type": "dir" if p.is_dir() else "file",
            "size_bytes": None if p.is_dir() else info.st_size,
            "modified": datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
        if len(entries) >= 200:
            truncated = True
            break
    return {"ok": True, "workspace": str(WORKSPACE), "folder": _rel(base),
            "count": len(entries), "truncated": truncated, "entries": entries}


def tool_read_workspace_file(path: str, max_chars: Any = 8000) -> dict:
    target = _safe_path(path)
    if not target.is_file():
        raise ValueError(f"File not found in workspace: {path}")
    limit = _as_int(max_chars, 8000, 1, 20000)
    with open(target, "rb") as fh:
        raw = fh.read(limit * 4 + 1)
    content = raw.decode("utf-8", "replace")
    return {"ok": True, "path": _rel(target), "size_bytes": target.stat().st_size,
            "truncated": len(content) > limit, "content": content[:limit]}


def tool_write_workspace_file(path: str, content: str = "", overwrite: Any = False) -> dict:
    target = _safe_path(path)
    if target == WORKSPACE or target.is_dir():
        raise ValueError("path must point to a file, not a folder.")
    if target.exists() and not _as_bool(overwrite):
        return {"ok": False, "error": "File already exists. Ask again with overwrite=true to replace it."}
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _text(content)
    target.write_text(data, encoding="utf-8")
    return {"ok": True, "path": _rel(target), "bytes_written": len(data.encode("utf-8"))}


def tool_delete_workspace_path(path: str) -> dict:
    target = _safe_path(path)
    if target == WORKSPACE:
        raise ValueError("Refusing to delete the workspace root.")
    if not target.exists():
        raise ValueError(f"Path not found: {path}")
    if target.is_dir():
        if any(target.iterdir()):
            raise ValueError("Directory is not empty; delete its contents first.")
        target.rmdir()
        return {"ok": True, "deleted": _rel(target), "type": "dir"}
    target.unlink()
    return {"ok": True, "deleted": _rel(target), "type": "file"}


def tool_move_workspace_path(source: str, destination: str) -> dict:
    src, dst = _safe_path(source), _safe_path(destination)
    if src == WORKSPACE or dst == WORKSPACE:
        raise ValueError("Cannot move or replace the workspace root.")
    if not src.exists():
        raise ValueError(f"Source not found: {source}")
    if dst.exists():
        raise ValueError("Destination already exists.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"ok": True, "from": _rel(src), "to": _rel(dst)}


# ---------- PC: applications & shell ----------------------------------------
APP_ALIASES: dict[str, Any] = {
    "calculator": {"Windows": "calc.exe", "Darwin": "Calculator",
                   "Linux": ["gnome-calculator", "kcalc", "galculator", "qalculate-gtk"]},
    "calc": {"Windows": "calc.exe", "Darwin": "Calculator",
             "Linux": ["gnome-calculator", "kcalc", "galculator"]},
    "notepad": {"Windows": "notepad.exe", "Darwin": "TextEdit", "Linux": ["gedit", "kate", "mousepad"]},
    "text editor": {"Windows": "notepad.exe", "Darwin": "TextEdit", "Linux": ["gedit", "kate", "mousepad"]},
    "file explorer": {"Windows": "explorer.exe", "Darwin": "Finder", "Linux": ["nautilus", "dolphin", "thunar", "nemo"]},
    "explorer": {"Windows": "explorer.exe", "Darwin": "Finder", "Linux": ["nautilus", "dolphin", "thunar", "nemo"]},
    "files": {"Windows": "explorer.exe", "Darwin": "Finder", "Linux": ["nautilus", "dolphin", "thunar", "nemo"]},
    "terminal": {"Windows": "cmd.exe", "Darwin": "Terminal",
                 "Linux": ["gnome-terminal", "konsole", "xfce4-terminal", "xterm"]},
    "task manager": {"Windows": "taskmgr.exe", "Darwin": "Activity Monitor",
                     "Linux": ["gnome-system-monitor", "ksysguard", "htop"]},
    "paint": {"Windows": "mspaint.exe", "Darwin": "Preview", "Linux": ["gimp", "kolourpaint"]},
    "browser": {"Windows": "msedge", "Darwin": "Safari", "Linux": ["firefox", "google-chrome", "chromium"]},
    "chrome": {"Windows": "chrome", "Darwin": "Google Chrome", "Linux": ["google-chrome", "chromium", "chrome"]},
    "firefox": {"Windows": "firefox", "Darwin": "Firefox", "Linux": ["firefox"]},
    "edge": {"Windows": "msedge", "Darwin": "Microsoft Edge", "Linux": ["microsoft-edge"]},
    "vscode": {"Windows": "code", "Darwin": "Visual Studio Code", "Linux": ["code"]},
    "spotify": {"Windows": "spotify", "Darwin": "Spotify", "Linux": ["spotify"]},
}


def _resolve_candidates(name: str) -> list[str]:
    entry = APP_ALIASES.get(name.lower().strip())
    if not entry:
        return [name]
    value = entry.get(SYSTEM_NAME, name)
    return [value] if isinstance(value, str) else list(value)


def tool_launch_application(app_name: str, arguments: str = "") -> dict:
    name = (app_name or "").strip()
    if not name:
        raise ValueError("app_name is required.")
    # No shell metacharacters (& | < > ^ " % ( ) ; ` $) are allowed in names.
    if not re.fullmatch(r"[\w .\-+\\/:]+", name):
        raise ValueError("Application name contains unsupported characters.")
    args = shlex.split(arguments, posix=not IS_WINDOWS) if arguments else []
    candidates = _resolve_candidates(name)

    if IS_MAC:
        cmd = ["open", "-a", candidates[0]] + (["--args", *args] if args else [])
        subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return {"ok": True, "launched": candidates[0], "args": args}

    for cand in candidates:
        exe = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if exe:
            proc = subprocess.Popen(
                [exe, *args], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=not IS_WINDOWS,
            )
            return {"ok": True, "launched": exe, "args": args, "pid": proc.pid}

    if IS_WINDOWS:
        # `start` resolves apps registered with Windows (chrome, spotify, ...).
        safe_arg = re.compile(r"[\w .\-+\\/:?=#@~,]+")
        if not all(safe_arg.fullmatch(a) for a in args):
            raise ValueError("Arguments contain characters that are not allowed for this launch method.")
        proc = subprocess.run(
            ["cmd", "/c", "start", "", candidates[0], *args],
            capture_output=True, text=True, errors="replace", timeout=10,
            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
        )
        if proc.returncode == 0:
            return {"ok": True, "launched": candidates[0], "args": args, "method": "windows-start"}
        return {"ok": False, "error": f"Could not find or start '{name}'.",
                "detail": _clip((proc.stderr or proc.stdout).strip(), 300)}

    return {"ok": False, "error": f"'{name}' was not found on PATH. Tried: {', '.join(candidates)}"}


def tool_run_shell_command(command: str, timeout_seconds: Any = 30, working_dir: str = "") -> dict:
    command = (command or "").strip()
    if not command:
        raise ValueError("command is required.")
    timeout = _as_int(timeout_seconds, 30, 1, 120)
    cwd = _safe_path(working_dir) if working_dir else WORKSPACE
    if not cwd.is_dir():
        raise ValueError("working_dir is not a directory inside the workspace.")
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(cwd), capture_output=True, text=True,
            errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
            creationflags=NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"Command timed out after {timeout}s.",
                "stdout": _clip(_text(exc.stdout).strip(), 1500),
                "stderr": _clip(_text(exc.stderr).strip(), 800)}
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "cwd": str(cwd),
            "stdout": _clip(proc.stdout.strip(), 3000), "stderr": _clip(proc.stderr.strip(), 1000)}


# ---------- Android: ADB bridge ----------------------------------------------
KEYCODES = {
    "home": 3, "back": 4, "recents": 187, "power": 26, "wake": 224, "sleep": 223,
    "volume_up": 24, "volume_down": 25, "mute": 164, "play_pause": 85,
    "next_track": 87, "previous_track": 88,
}
ADB_ACTIONS = [
    "device_info", "battery", "screenshot", *KEYCODES.keys(), "open_url",
    "launch_app", "tap", "swipe", "type_text", "list_packages",
]


def _adb_exec(args: list[str], serial: str = "", timeout: int = 30) -> subprocess.CompletedProcess:
    adb = shutil.which("adb")
    if not adb:
        raise RuntimeError("ADB not found on PATH. Install Android platform-tools and enable USB debugging.")
    cmd = [adb]
    if serial:
        if not re.fullmatch(r"[\w.:\-]+", serial):
            raise ValueError("Invalid device serial.")
        cmd += ["-s", serial]
    cmd += args
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ADB command timed out after {timeout}s.")


def _adb_shell(shell_cmd: str, serial: str = "", timeout: int = 30) -> dict:
    proc = _adb_exec(["shell", shell_cmd], serial, timeout)
    return {"ok": proc.returncode == 0, "returncode": proc.returncode,
            "stdout": _clip(_text(proc.stdout).strip(), 3000),
            "stderr": _clip(_text(proc.stderr).strip(), 800)}


def tool_adb_list_devices() -> dict:
    proc = _adb_exec(["devices", "-l"])
    devices = []
    for line in _text(proc.stdout).splitlines():
        line = line.strip()
        if not line or line.startswith("*") or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            devices.append({"serial": parts[0], "state": parts[1], "details": " ".join(parts[2:])})
    result: dict[str, Any] = {"ok": proc.returncode == 0, "device_count": len(devices), "devices": devices}
    if any(d["state"] == "unauthorized" for d in devices):
        result["hint"] = "A device is unauthorized - accept the USB debugging prompt on the phone."
    if not devices:
        result["hint"] = "No devices found. Check the cable, enable USB debugging, and authorize this PC."
    if proc.returncode != 0:
        result["stderr"] = _clip(_text(proc.stderr).strip(), 500)
    return result


def tool_adb_shell(command: str, serial: str = "", timeout_seconds: Any = 30) -> dict:
    command = (command or "").strip()
    if not command:
        raise ValueError("command is required.")
    return _adb_shell(command, (serial or "").strip(), _as_int(timeout_seconds, 30, 1, 120))


def tool_adb_phone_action(
    action: str, serial: str = "", text: str = "", url: str = "", package: str = "",
    x: Any = None, y: Any = None, x2: Any = None, y2: Any = None, duration_ms: Any = 300,
) -> dict:
    action = (action or "").strip().lower()
    serial = (serial or "").strip()

    if action in KEYCODES:
        result = _adb_shell(f"input keyevent {KEYCODES[action]}", serial)
    elif action == "device_info":
        result = _adb_shell(
            "getprop ro.product.manufacturer; getprop ro.product.model; "
            "getprop ro.build.version.release; getprop ro.build.version.sdk", serial)
        lines = result.get("stdout", "").splitlines()
        if len(lines) >= 4:
            result.update(manufacturer=lines[0], model=lines[1], android_version=lines[2], sdk=lines[3])
    elif action == "battery":
        result = _adb_shell("dumpsys battery", serial)
        info: dict[str, Any] = {}
        wanted = {"level", "status", "health", "AC powered", "USB powered", "Wireless powered", "temperature", "voltage"}
        for line in result.get("stdout", "").splitlines():
            if ":" in line:
                key, val = (s.strip() for s in line.split(":", 1))
                if key in wanted:
                    info[key] = val
        if info.get("temperature", "").lstrip("-").isdigit():
            info["temperature_c"] = int(info["temperature"]) / 10
        info["status_text"] = {"2": "charging", "3": "discharging", "4": "not charging", "5": "full"}.get(
            info.get("status", ""), "unknown")
        result = {"ok": result["ok"], "battery": info}
    elif action == "screenshot":
        proc = _adb_exec(["exec-out", "screencap", "-p"], serial, timeout=30)
        if proc.returncode != 0 or not proc.stdout.startswith(b"\x89PNG"):
            return {"ok": False, "error": "Screenshot failed.", "stderr": _clip(_text(proc.stderr).strip(), 400)}
        folder = WORKSPACE / "phone_screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
        target.write_bytes(proc.stdout)
        result = {"ok": True, "saved_to": _rel(target), "bytes": len(proc.stdout)}
    elif action == "open_url":
        if not re.fullmatch(r"https?://\S+", url or ""):
            raise ValueError("url must start with http:// or https://")
        result = _adb_shell(f"am start -a android.intent.action.VIEW -d {shlex.quote(url)}", serial)
    elif action == "launch_app":
        if not re.fullmatch(r"[A-Za-z0-9_.]+", package or ""):
            raise ValueError("package must look like com.example.app")
        result = _adb_shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1", serial)
    elif action == "tap":
        result = _adb_shell(f"input tap {_req_int(x, 'x')} {_req_int(y, 'y')}", serial)
    elif action == "swipe":
        dur = _as_int(duration_ms, 300, 50, 5000)
        result = _adb_shell(
            f"input swipe {_req_int(x, 'x')} {_req_int(y, 'y')} {_req_int(x2, 'x2')} {_req_int(y2, 'y2')} {dur}",
            serial)
    elif action == "type_text":
        if not text:
            raise ValueError("text is required.")
        escaped = shlex.quote(text[:200].replace(" ", "%s"))
        result = _adb_shell(f"input text {escaped}", serial)
    elif action == "list_packages":
        result = _adb_shell("pm list packages -3", serial)
        pkgs = [ln.replace("package:", "").strip() for ln in result.get("stdout", "").splitlines() if ln.strip()]
        result = {"ok": result["ok"], "count": len(pkgs), "third_party_packages": pkgs[:150]}
    else:
        raise ValueError(f"Unknown action '{action}'. Valid: {', '.join(ADB_ACTIONS)}")

    result["action"] = action
    return result


# ======================================================================
# Tool registry (schemas sent to Groq + metadata for the safety gate)
# ======================================================================
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., dict]
    sensitive: bool                      # True -> needs explicit user approval
    risk: str                            # "low" | "medium" | "high"
    summarize: Callable[[dict], str]     # human-readable preview for the gate


def _schema(props: Optional[dict] = None, required: Optional[list] = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


_STR = {"type": "string"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}

TOOLS: list[ToolSpec] = [
    ToolSpec(
        "get_system_stats",
        "Read live PC metrics: CPU, RAM, disks, battery, uptime, top processes.",
        _schema(), tool_get_system_stats, False, "low",
        lambda a: "Read CPU / RAM / disk / battery telemetry",
    ),
    ToolSpec(
        "list_workspace_files",
        "List files and folders inside the user's JARVIS workspace folder.",
        _schema({"subdir": {**_STR, "description": "Sub-folder, relative to workspace. Empty = root."},
                 "recursive": {**_BOOL, "description": "List nested contents too."}}),
        tool_list_workspace_files, False, "low",
        lambda a: f"List workspace folder '{a.get('subdir') or '.'}'",
    ),
    ToolSpec(
        "read_workspace_file",
        "Read a text file from the workspace.",
        _schema({"path": {**_STR, "description": "File path relative to workspace."},
                 "max_chars": {**_INT, "description": "Max characters to return (default 8000)."}}, ["path"]),
        tool_read_workspace_file, False, "low",
        lambda a: f"Read workspace file '{a.get('path')}'",
    ),
    ToolSpec(
        "write_workspace_file",
        "Create or overwrite a text file in the workspace.",
        _schema({"path": {**_STR, "description": "File path relative to workspace."},
                 "content": {**_STR, "description": "Full text content to write."},
                 "overwrite": {**_BOOL, "description": "Replace the file if it exists."}}, ["path", "content"]),
        tool_write_workspace_file, True, "medium",
        lambda a: (f"WRITE workspace file '{a.get('path')}' (overwrite={_as_bool(a.get('overwrite'))})\n"
                   f"--- content preview ---\n{_clip(_text(a.get('content')), 600)}"),
    ),
    ToolSpec(
        "delete_workspace_path",
        "Delete one file or one empty folder inside the workspace.",
        _schema({"path": {**_STR, "description": "Path relative to workspace."}}, ["path"]),
        tool_delete_workspace_path, True, "high",
        lambda a: f"DELETE '{a.get('path')}' from the workspace (irreversible)",
    ),
    ToolSpec(
        "move_workspace_path",
        "Move or rename a file/folder inside the workspace.",
        _schema({"source": {**_STR, "description": "Existing path relative to workspace."},
                 "destination": {**_STR, "description": "New path relative to workspace."}},
                ["source", "destination"]),
        tool_move_workspace_path, True, "medium",
        lambda a: f"MOVE '{a.get('source')}'  ->  '{a.get('destination')}'",
    ),
    ToolSpec(
        "launch_application",
        "Launch a desktop application on the PC (e.g. calculator, notepad, chrome, vscode).",
        _schema({"app_name": {**_STR, "description": "App name or executable, e.g. 'calculator', 'chrome'."},
                 "arguments": {**_STR, "description": "Optional arguments, e.g. a URL or file path."}}, ["app_name"]),
        tool_launch_application, True, "medium",
        lambda a: f"LAUNCH application: {a.get('app_name')} {a.get('arguments') or ''}".strip(),
    ),
    ToolSpec(
        "run_shell_command",
        "Run a shell command on the PC (cwd = workspace). Use only when no dedicated tool fits.",
        _schema({"command": {**_STR, "description": "The exact command line to execute."},
                 "timeout_seconds": {**_INT, "description": "Timeout 1-120 (default 30)."},
                 "working_dir": {**_STR, "description": "Optional workspace sub-folder to run in."}}, ["command"]),
        tool_run_shell_command, True, "high",
        lambda a: (f"$ {a.get('command')}\n"
                   f"(cwd: {a.get('working_dir') or WORKSPACE}, timeout: {_as_int(a.get('timeout_seconds'), 30, 1, 120)}s)\n"
                   "NOTE: the command itself is NOT sandboxed to the workspace."),
    ),
    ToolSpec(
        "adb_list_devices",
        "List Android devices connected through ADB (runs `adb devices -l`).",
        _schema(), tool_adb_list_devices, True, "medium",
        lambda a: "$ adb devices -l",
    ),
    ToolSpec(
        "adb_phone_action",
        ("Control a connected Android phone with a preset action: device_info, battery, screenshot, "
         "home, back, recents, power, wake, sleep, volume_up, volume_down, mute, play_pause, next_track, "
         "previous_track, open_url, launch_app, tap, swipe, type_text, list_packages."),
        _schema({
            "action": {**_STR, "enum": ADB_ACTIONS},
            "serial": {**_STR, "description": "Device serial (optional if one device)."},
            "text": {**_STR, "description": "For type_text."},
            "url": {**_STR, "description": "For open_url (http/https)."},
            "package": {**_STR, "description": "For launch_app, e.g. com.whatsapp."},
            "x": {**_INT, "description": "tap/swipe start X."}, "y": {**_INT, "description": "tap/swipe start Y."},
            "x2": {**_INT, "description": "swipe end X."}, "y2": {**_INT, "description": "swipe end Y."},
            "duration_ms": {**_INT, "description": "swipe duration (default 300)."},
        }, ["action"]),
        tool_adb_phone_action, True, "medium",
        lambda a: ("$ adb " + (f"-s {a['serial']} " if a.get("serial") else "") +
                   f"[phone action: {a.get('action')}] " +
                   " ".join(f"{k}={a[k]!r}" for k in ("text", "url", "package", "x", "y", "x2", "y2") if a.get(k) not in (None, ""))),
    ),
    ToolSpec(
        "adb_shell",
        "Run an arbitrary `adb shell` command on the Android phone. Prefer adb_phone_action when possible.",
        _schema({"command": {**_STR, "description": "Command executed on the phone's shell."},
                 "serial": {**_STR, "description": "Device serial (optional if one device)."},
                 "timeout_seconds": {**_INT, "description": "Timeout 1-120 (default 30)."}}, ["command"]),
        tool_adb_shell, True, "high",
        lambda a: "$ adb " + (f"-s {a['serial']} " if a.get("serial") else "") + f"shell {a.get('command')}",
    ),
]

TOOL_REGISTRY: dict[str, ToolSpec] = {t.name: t for t in TOOLS}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
    for t in TOOLS
]


def parse_args(raw: Optional[str]) -> tuple[dict, Optional[str]]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"Malformed JSON arguments: {exc}"
    if not isinstance(data, dict):
        return {}, "Arguments must be a JSON object."
    return {k: v for k, v in data.items() if v is not None}, None


def execute_tool_call(call: dict) -> str:
    """Run a tool call and return a JSON string for the model. Never raises."""
    spec = TOOL_REGISTRY.get(call["name"])
    if spec is None:
        return json.dumps({"ok": False, "error": f"Unknown tool '{call['name']}'."})
    args, err = parse_args(call.get("arguments"))
    if err:
        return json.dumps({"ok": False, "error": err})
    accepted = inspect.signature(spec.handler).parameters
    args = {k: v for k, v in args.items() if k in accepted}
    try:
        result = spec.handler(**args)
    except TypeError as exc:
        result = {"ok": False, "error": f"Invalid arguments: {exc}"}
    except subprocess.TimeoutExpired as exc:
        result = {"ok": False, "error": f"Timed out after {exc.timeout}s."}
    except Exception as exc:  # noqa: BLE001 - surface every failure to the model
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return _clip(json.dumps(result, default=str, ensure_ascii=False), MAX_TOOL_OUTPUT)


# ======================================================================
# AI brain (Groq) + agent loop with the confirmation gate
# ======================================================================
def build_system_prompt() -> str:
    adb = "available" if shutil.which("adb") else "NOT installed on this PC"
    return (
        f"You are J.A.R.V.I.S., a refined, quietly witty personal AI assistant in the spirit of the Stark "
        f"Industries system. Address the user as \"{get_title()}\". Be concise, precise and helpful.\n\n"
        f"ENVIRONMENT: OS={platform.platform()}; local time={datetime.now():%Y-%m-%d %H:%M}; "
        f"workspace={WORKSPACE} (all file tools are sandboxed to it); ADB is {adb}.\n\n"
        "RULES:\n"
        "- Use tools for live data or real-world actions. Never invent tool output.\n"
        "- Any tool that runs a command, launches software, modifies files, or contacts the phone needs the "
        "user's approval in the interface. Simply call the tool and wait for the result.\n"
        "- If a result contains DENIED_BY_USER, acknowledge it gracefully, do not retry the same call, and "
        "offer an alternative.\n"
        "- Report only what tool results confirm. If a tool fails, explain plainly and suggest a fix.\n"
        "- Uploaded documents appear inside [[ATTACHMENT:path]] ... [[/ATTACHMENT]] blocks; use them directly.\n"
        "- Prefer dedicated tools over run_shell_command. Keep replies short unless detail is requested."
    )


def trim_history(msgs: list[dict]) -> list[dict]:
    """Last MAX_HISTORY messages, starting on a user turn so tool pairs never break."""
    start = max(0, len(msgs) - MAX_HISTORY)
    while start < len(msgs) and msgs[start]["role"] != "user":
        start += 1
    if start >= len(msgs):
        start = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    return msgs[start:]


def call_model(client: Any) -> Any:
    payload = [{"role": "system", "content": build_system_prompt()}] + trim_history(ss.messages)
    last_exc: Optional[Exception] = None
    for _ in range(2):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=payload, tools=TOOL_SCHEMAS, tool_choice="auto",
                temperature=0.3, max_tokens=1500,
            )
        except groq_sdk.BadRequestError as exc:
            # Llama occasionally emits a malformed tool call; one retry usually fixes it.
            last_exc = exc
            if "tool_use_failed" in str(exc) or "failed_generation" in str(exc):
                continue
            raise
    assert last_exc is not None
    raise last_exc


def agent_loop() -> None:
    """Drive model <-> tools until an answer is produced or approval is needed."""
    ss.last_error, ss.can_retry = None, False
    key = get_api_key()
    if not key:
        ss.last_error = "No Groq API key set. Paste it in the sidebar."
        return
    client = Groq(api_key=key)
    model_calls = 0

    while True:
        # 1) Execute queued tool calls; stop at the first one that needs approval.
        while ss.tool_queue:
            call = ss.tool_queue[0]
            spec = TOOL_REGISTRY.get(call["name"])
            if spec is not None and spec.sensitive:
                ss.awaiting = call
                audit("PENDING", call["name"], _clip(call["arguments"], 120))
                return
            result = execute_tool_call(call)
            audit("AUTO", call["name"], _clip(call["arguments"], 120))
            ss.messages.append({"role": "tool", "tool_call_id": call["id"],
                                "name": call["name"], "content": result})
            ss.tool_queue.pop(0)

        # 2) Ask the model what to do next.
        if model_calls >= MAX_AGENT_STEPS:
            ss.last_error = "Step limit reached for this request. Send another message to continue."
            return
        model_calls += 1
        try:
            resp = call_model(client)
        except groq_sdk.AuthenticationError:
            ss.last_error = "Groq rejected the API key. Check it in the sidebar."
            return
        except groq_sdk.RateLimitError:
            ss.last_error = "Groq rate limit hit (free tier). Wait a few seconds and retry."
            ss.can_retry = True
            return
        except groq_sdk.APIConnectionError:
            ss.last_error = "Cannot reach Groq. Check your internet connection."
            ss.can_retry = True
            return
        except groq_sdk.APIStatusError as exc:
            ss.last_error = f"Groq API error ({exc.status_code}): {_clip(str(exc), 300)}"
            ss.can_retry = True
            return
        except Exception as exc:  # noqa: BLE001
            ss.last_error = f"Unexpected error: {type(exc).__name__}: {_clip(str(exc), 300)}"
            ss.can_retry = True
            return

        msg = resp.choices[0].message
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                for tc in msg.tool_calls
            ]
        ss.messages.append(entry)
        if not msg.tool_calls:
            return
        ss.tool_queue.extend(
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
            for tc in msg.tool_calls
        )


def resolve_gate(approved: bool) -> None:
    call = ss.awaiting
    ss.awaiting = None
    if not call:
        return
    if ss.tool_queue and ss.tool_queue[0]["id"] == call["id"]:
        ss.tool_queue.pop(0)
    if approved:
        result = execute_tool_call(call)
        audit("APPROVED", call["name"], _clip(call["arguments"], 120))
    else:
        result = json.dumps({"ok": False, "error": "DENIED_BY_USER: the user declined to authorize this action. "
                                                   "Do not retry it."})
        audit("DENIED", call["name"], _clip(call["arguments"], 120))
    ss.messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result})
    agent_loop()


def reset_conversation() -> None:
    ss.messages, ss.tool_queue, ss.attachments = [], [], []
    ss.awaiting, ss.last_error, ss.can_retry = None, None, False
    audit("RESET", "conversation")


# ======================================================================
# Uploads
# ======================================================================
def extract_text(name: str, data: bytes) -> Optional[str]:
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXTS or ext == "":
        return data.decode("utf-8", "replace")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # optional dependency
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages[:30])
        except Exception:
            return None
    return None


def _unique_path(directory: Path, name: str) -> Path:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = directory / name, 1
    while candidate.exists():
        candidate = directory / f"{stem}_{n}{suffix}"
        n += 1
    return candidate


def ingest_files(files: list) -> tuple[int, list[str]]:
    folder = WORKSPACE / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    done, problems = 0, []
    for f in files:
        data = f.getvalue()
        if len(data) > MAX_UPLOAD_BYTES:
            problems.append(f"{f.name}: larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB, skipped")
            continue
        safe_name = re.sub(r"[^\w.\- ]", "_", Path(f.name).name) or "upload.bin"
        target = _unique_path(folder, safe_name)
        target.write_bytes(data)
        text = extract_text(safe_name, data)
        if text is not None and len(text) > MAX_ATTACH_CHARS:
            text = text[:MAX_ATTACH_CHARS] + f"\n... [truncated, full file saved at {_rel(target)}]"
        ss.attachments.append({"path": _rel(target), "text": text})
        audit("UPLOAD", _rel(target), f"{len(data)} bytes")
        done += 1
    return done, problems


ATTACH_RE = re.compile(r"\[\[ATTACHMENT:(.*?)\]\]\n(.*?)\n\[\[/ATTACHMENT\]\]", re.DOTALL)


def compose_user_content(prompt: str) -> str:
    blocks = []
    for att in ss.attachments:
        body = att["text"] if att["text"] is not None else (
            "(binary or unsupported file type - saved to the workspace, content not readable here)")
        blocks.append(f"[[ATTACHMENT:{att['path']}]]\n{body or '(empty file)'}\n[[/ATTACHMENT]]")
    ss.attachments = []
    return "\n\n".join(blocks + [prompt])


def split_attachments(content: str) -> tuple[str, list[tuple[str, str]]]:
    found = [(m.group(1), m.group(2)) for m in ATTACH_RE.finditer(content)]
    return ATTACH_RE.sub("", content).strip(), found


# ======================================================================
# Styling
# ======================================================================
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600&family=Share+Tech+Mono&display=swap');
:root{--cyan:#00e5ff;--cyan-dim:#0aa7c2;--amber:#ffb300;--red:#ff3d57;--green:#3dff9a;--text:#cfefff;}
html,body,.stApp,[data-testid="stAppViewContainer"]{font-family:'Rajdhani','Segoe UI',sans-serif;}
.stApp{background:
 radial-gradient(1100px 600px at 75% -10%,rgba(0,229,255,.10),transparent 60%),
 radial-gradient(900px 500px at -10% 110%,rgba(0,110,255,.09),transparent 60%),#04070d;color:var(--text);}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
 background-image:linear-gradient(rgba(0,229,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.035) 1px,transparent 1px);
 background-size:42px 42px;}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2rem;max-width:1180px;}
.stMarkdown p,.stMarkdown li{font-size:1.06rem;line-height:1.5;color:var(--text);}
code,pre,.stCode{font-family:'Share Tech Mono',monospace !important;}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:rgba(0,229,255,.35);border-radius:4px}

section[data-testid="stSidebar"]{background:linear-gradient(180deg,#050c18,#030710);border-right:1px solid rgba(0,229,255,.28);}
.side-brand{font-family:'Orbitron',sans-serif;font-weight:900;letter-spacing:.18em;color:var(--cyan);font-size:1rem;text-shadow:0 0 10px rgba(0,229,255,.7);margin-bottom:.4rem;}
.side-h{font-family:'Orbitron',sans-serif;font-size:.72rem;letter-spacing:.22em;color:var(--cyan-dim);border-bottom:1px solid rgba(0,229,255,.25);padding:.9rem 0 .3rem;margin-bottom:.6rem;}
section[data-testid="stSidebar"] label p{font-family:'Share Tech Mono',monospace;font-size:.72rem !important;letter-spacing:.1em;color:#7fdcef !important;}

.stTextInput input{background:#030a14 !important;border:1px solid rgba(0,229,255,.35) !important;color:#aef3ff !important;font-family:'Share Tech Mono',monospace !important;}
.stTextInput input:focus{border-color:var(--cyan) !important;box-shadow:0 0 10px rgba(0,229,255,.4) !important;}

.stButton>button{width:100%;background:rgba(0,229,255,.06);border:1px solid rgba(0,229,255,.45);color:#aef3ff;border-radius:4px;
 font-family:'Share Tech Mono',monospace;letter-spacing:.05em;text-transform:uppercase;font-size:.78rem;transition:all .2s;}
.stButton>button:hover:not(:disabled){background:rgba(0,229,255,.16);border-color:var(--cyan);box-shadow:0 0 14px rgba(0,229,255,.45);color:#fff;}
.stButton>button:disabled{opacity:.35;}
[data-testid="stBaseButton-primary"]{border-color:var(--green) !important;color:var(--green) !important;background:rgba(61,255,154,.08) !important;}
[data-testid="stBaseButton-primary"]:hover{box-shadow:0 0 14px rgba(61,255,154,.5) !important;background:rgba(61,255,154,.18) !important;}
.st-key-gate_no button{border-color:var(--red) !important;color:#ff8a9a !important;background:rgba(255,61,87,.08) !important;}
.st-key-gate_no button:hover{box-shadow:0 0 14px rgba(255,61,87,.5) !important;}

[data-testid="stChatMessage"]{background:rgba(8,22,40,.68);border:1px solid rgba(0,229,255,.16);border-left:3px solid var(--cyan);border-radius:6px;}
[data-testid="stChatInput"]{border:1px solid rgba(0,229,255,.5);border-radius:6px;background:rgba(4,12,24,.92);}
[data-testid="stChatInput"] textarea{color:#cfefff !important;font-family:'Share Tech Mono',monospace !important;}
[data-testid="stExpander"]{border:1px solid rgba(0,229,255,.2) !important;background:rgba(4,14,26,.6);border-radius:6px;}
[data-testid="stExpander"] summary p{font-family:'Share Tech Mono',monospace;font-size:.82rem;color:#8eeeff;}

.jv-header{display:flex;align-items:center;gap:22px;margin-bottom:.4rem;}
.jv-title{font-family:'Orbitron',sans-serif;font-weight:900;font-size:2.1rem;letter-spacing:.38em;color:var(--cyan);text-shadow:0 0 18px rgba(0,229,255,.75);}
.jv-sub{font-family:'Share Tech Mono',monospace;font-size:.78rem;letter-spacing:.3em;color:#58bcd0;}
.arc{position:relative;width:76px;height:76px;flex:0 0 76px;}
.arc .ring{position:absolute;border-radius:50%;border:2px solid transparent;}
.arc .r1{inset:0;border-top-color:var(--cyan);border-bottom-color:var(--cyan);animation:spin 4s linear infinite;}
.arc .r2{inset:9px;border-left-color:#00b8d4;border-right-color:#00b8d4;animation:spin 3s linear infinite reverse;}
.arc .r3{inset:18px;border:2px dashed rgba(0,229,255,.6);animation:spin 9s linear infinite;}
.arc .core{position:absolute;inset:28px;border-radius:50%;background:radial-gradient(circle,#e6fdff 0%,#00e5ff 45%,rgba(0,229,255,0) 72%);
 box-shadow:0 0 18px #00e5ff,0 0 42px rgba(0,229,255,.6);animation:pulse 2.2s ease-in-out infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{0%,100%{opacity:.7}50%{opacity:1}}
.chips{margin:.5rem 0 1.1rem;}
.chip{display:inline-block;padding:3px 10px;margin:0 6px 6px 0;border:1px solid rgba(0,229,255,.35);border-radius:3px;
 font-family:'Share Tech Mono',monospace;font-size:.74rem;letter-spacing:.06em;color:#7fdcef;background:rgba(0,229,255,.05);}
.chip b{color:var(--cyan);font-weight:400;}
.chip.ok b{color:var(--green);} .chip.warn b{color:var(--amber);} .chip.bad b{color:var(--red);}

.welcome{border:1px solid rgba(0,229,255,.3);background:linear-gradient(135deg,rgba(0,229,255,.07),rgba(0,60,120,.10));
 border-radius:8px;padding:18px 22px;margin:.6rem 0 1rem;font-size:1.1rem;}
.welcome .w-h{font-family:'Orbitron',sans-serif;letter-spacing:.2em;color:var(--cyan);font-size:.85rem;margin-bottom:.4rem;}

.gate{border:1px solid var(--amber);background:rgba(255,179,0,.07);color:#ffcf5a;padding:10px 14px;border-radius:6px;margin:.6rem 0 .4rem;
 font-family:'Orbitron',sans-serif;font-size:.82rem;letter-spacing:.14em;animation:gatepulse 1.6s ease-in-out infinite;}
.gate.high{border-color:var(--red);background:rgba(255,61,87,.08);color:#ff8a9a;}
@keyframes gatepulse{0%,100%{box-shadow:0 0 0 0 transparent}50%{box-shadow:0 0 18px currentColor}}

.stat{margin-bottom:.7rem;}
.stat-h{display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:.72rem;letter-spacing:.08em;color:#8eeeff;margin-bottom:3px;}
.track{height:7px;background:rgba(0,229,255,.12);border-radius:4px;overflow:hidden;}
.fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#007c91,#00e5ff);box-shadow:0 0 8px #00e5ff;transition:width .6s;}
.fill.warn{background:linear-gradient(90deg,#9a6400,#ffb300);box-shadow:0 0 8px #ffb300;}
.fill.bad{background:linear-gradient(90deg,#a0132a,#ff3d57);box-shadow:0 0 8px #ff3d57;}
</style>"""


# ======================================================================
# UI components
# ======================================================================
def _fragment(run_every: float) -> Callable:
    """st.fragment(run_every=...) with a graceful fallback for old Streamlit."""
    def decorator(fn: Callable) -> Callable:
        frag = getattr(st, "fragment", None)
        if frag is None:
            return fn
        try:
            return frag(run_every=run_every)(fn)
        except TypeError:
            return fn
    return decorator


def _bar(label: str, pct: float, detail: str) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    state = "bad" if pct >= 90 else "warn" if pct >= 70 else ""
    return (f'<div class="stat"><div class="stat-h"><span>{html.escape(label)}</span>'
            f'<span>{html.escape(detail)}</span></div>'
            f'<div class="track"><div class="fill {state}" style="width:{pct:.0f}%"></div></div></div>')


@_fragment(run_every=2)
def live_stats() -> None:
    try:
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        du = psutil.disk_usage(DISK_ROOT)
        uptime = _fmt_uptime(time.time() - psutil.boot_time())
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Telemetry unavailable: {exc}")
        return
    parts = [
        _bar("CPU", cpu, f"{cpu:.0f}% / {psutil.cpu_count()} threads"),
        _bar("MEMORY", vm.percent, f"{vm.used / GB:.1f} / {vm.total / GB:.1f} GB"),
        _bar("DISK " + DISK_ROOT.rstrip("\\/") or "/", du.percent, f"{du.used / GB:.0f} / {du.total / GB:.0f} GB"),
    ]
    try:
        battery = psutil.sensors_battery()
    except Exception:
        battery = None
    if battery:
        parts.append(_bar("BATTERY" + (" (AC)" if battery.power_plugged else ""),
                          battery.percent, f"{battery.percent:.0f}%"))
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.caption(f"UPTIME {uptime}  |  {time.strftime('%H:%M:%S')}")


def _chip(label: str, value: str, state: str = "") -> str:
    return f'<span class="chip {state}">{html.escape(label)}: <b>{html.escape(value)}</b></span>'


def render_header() -> None:
    has_key = bool(get_api_key())
    chips = "".join([
        _chip("CORE", MODEL),
        _chip("GROQ KEY", "LOADED" if has_key else "MISSING", "ok" if has_key else "bad"),
        _chip("HOST", f"{SYSTEM_NAME} / {platform.node()[:18]}"),
        _chip("ADB", "FOUND" if shutil.which("adb") else "NOT FOUND", "ok" if shutil.which("adb") else "warn"),
        _chip("SAFETY GATE", "ARMED", "ok"),
    ])
    st.markdown(
        '<div class="jv-header"><div class="arc"><div class="ring r1"></div><div class="ring r2"></div>'
        '<div class="ring r3"></div><div class="core"></div></div>'
        '<div><div class="jv-title">J.A.R.V.I.S.</div>'
        '<div class="jv-sub">JUST A RATHER VERY INTELLIGENT SYSTEM</div></div></div>'
        f'<div class="chips">{chips}</div>',
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    title = html.escape(get_title())
    if get_api_key():
        body = (f"Good day, {title}. All systems nominal and the safety gate is armed. "
                "Awaiting your instructions.")
    else:
        body = (f"Good day, {title}. My cognitive core is offline - paste your free Groq API key into the "
                "sidebar to bring me online.")
    st.markdown(f'<div class="welcome"><div class="w-h">&#9670; SYSTEM ONLINE</div>{body}</div>',
                unsafe_allow_html=True)


def _pretty(raw: str, limit: int = 2500) -> str:
    try:
        return _clip(json.dumps(json.loads(raw), indent=2, ensure_ascii=False), limit)
    except (TypeError, ValueError):
        return _clip(raw or "", limit)


def render_history() -> None:
    if not ss.messages:
        render_welcome()
        return
    for msg in ss.messages:
        role = msg["role"]
        if role == "user":
            text, atts = split_attachments(msg["content"])
            with st.chat_message("user", avatar="👤"):
                if text:
                    st.markdown(text)
                for name, body in atts:
                    with st.expander(f"📎 {name}"):
                        st.code(_clip(body, 2000), language="text")
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                if msg.get("content"):
                    st.markdown(msg["content"])
                for tc in msg.get("tool_calls", []):
                    fn = tc["function"]
                    with st.expander(f"⚙ TOOL CALL → {fn['name']}"):
                        st.code(_pretty(fn["arguments"], 1500), language="json")
        elif role == "tool":
            denied = "DENIED_BY_USER" in msg["content"]
            failed = '"ok": false' in msg["content"][:40]
            icon = "⛔" if denied else "⚠" if failed else "✔"
            with st.chat_message("assistant", avatar="🛠️"):
                with st.expander(f"{icon} RESULT ← {msg.get('name', 'tool')}"):
                    st.code(_pretty(msg["content"]), language="json")


def render_gate() -> None:
    call = ss.awaiting
    if not call:
        return
    spec = TOOL_REGISTRY.get(call["name"])
    args, err = parse_args(call["arguments"])
    risk = spec.risk if spec else "high"
    summary = spec.summarize(args) if (spec and not err) else f"Unparsed request: {call['arguments']}"

    st.markdown(
        f'<div class="gate {"high" if risk == "high" else ""}">&#9888; AUTHORIZATION REQUIRED &mdash; '
        f'{risk.upper()} RISK &middot; {html.escape(call["name"])}</div>',
        unsafe_allow_html=True,
    )
    st.code(summary, language="bash")
    with st.expander("Raw arguments"):
        st.code(json.dumps(args, indent=2, ensure_ascii=False), language="json")
    if risk == "high":
        st.caption("Review carefully - this action may be irreversible or reach beyond the workspace.")

    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("✅ APPROVE & EXECUTE", key="gate_ok", type="primary"):
        with st.spinner("Executing..."):
            resolve_gate(True)
        st.rerun()
    if c2.button("⛔ DENY", key="gate_no"):
        with st.spinner("Standing down..."):
            resolve_gate(False)
        st.rerun()


QUICK_ACTIONS = [
    ("📊 Diagnostics", "Run a full diagnostic of this PC with get_system_stats and give me a brief summary."),
    ("🗂 Workspace", "List the files in my workspace."),
    ("📱 Phone link", "Check which Android devices are connected via ADB and report their status."),
    ("🔋 Phone battery", "Get the battery status of my connected Android phone."),
    ("📸 Screenshot", "Take a screenshot of my Android phone and save it to my workspace."),
    ("🧮 Calculator", "Launch the calculator application."),
]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="side-brand">◢ J.A.R.V.I.S. CONSOLE</div>', unsafe_allow_html=True)

        # --- Authentication
        st.markdown('<div class="side-h">AUTHENTICATION</div>', unsafe_allow_html=True)
        st.text_input("GROQ API KEY", type="password", key="groq_api_key", placeholder="gsk_...",
                      help="Free key at https://console.groq.com/keys. Held in this session's memory only; "
                           "never written to disk.")
        st.text_input("ADDRESS ME AS", key="user_title")
        if st.button("VERIFY LINK", key="btn_verify"):
            if not get_api_key():
                ss.key_status = ("bad", "Enter a key first.")
            else:
                try:
                    Groq(api_key=get_api_key()).models.list()
                    ss.key_status = ("ok", "Link established. Core online.")
                except groq_sdk.AuthenticationError:
                    ss.key_status = ("bad", "Key rejected by Groq.")
                except Exception as exc:  # noqa: BLE001
                    ss.key_status = ("bad", _clip(f"{type(exc).__name__}: {exc}", 120))
        if ss.key_status:
            (st.success if ss.key_status[0] == "ok" else st.error)(ss.key_status[1])

        # --- Live telemetry
        st.markdown('<div class="side-h">SYSTEM STATUS</div>', unsafe_allow_html=True)
        live_stats()

        # --- Quick actions
        st.markdown('<div class="side-h">QUICK ACTIONS</div>', unsafe_allow_html=True)
        disabled = (not get_api_key()) or ss.awaiting is not None
        cols = st.columns(2)
        for i, (label, prompt) in enumerate(QUICK_ACTIONS):
            if cols[i % 2].button(label, key=f"qa_{i}", disabled=disabled):
                ss.quick_prompt = prompt
        st.caption("Anything that touches your system asks for approval first.")
        if st.button("🧹 CLEAR CONVERSATION", key="btn_clear"):
            reset_conversation()
            st.rerun()

        # --- Uploads
        st.markdown('<div class="side-h">DATA INGEST</div>', unsafe_allow_html=True)
        files = st.file_uploader("UPLOAD FILES / DOCUMENTS", accept_multiple_files=True,
                                 key=f"uploader_{ss.uploader_nonce}")
        if files and st.button("INGEST SELECTED FILES", key="btn_ingest"):
            count, problems = ingest_files(files)
            ss.uploader_nonce += 1
            ss.flash = f"Ingested {count} file(s) into the workspace." + (
                " Issues: " + "; ".join(problems) if problems else "")
            st.rerun()
        if ss.attachments:
            st.caption("ATTACHED TO NEXT MESSAGE:")
            for att in ss.attachments:
                st.markdown(f"`{att['path']}`" + ("" if att["text"] is not None else "  *(not readable)*"))
            if st.button("DETACH ALL", key="btn_detach"):
                ss.attachments = []
                st.rerun()

        # --- Workspace & audit
        with st.expander("🗂 WORKSPACE"):
            st.caption(str(WORKSPACE))
            try:
                items = sorted(WORKSPACE.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                lines = [("📁 " if p.is_dir() else "📄 ") + p.name for p in items if not p.name.startswith(".")][:40]
            except OSError:
                lines = []
            st.code("\n".join(lines) or "(empty)", language="text")
        with st.expander("🧾 AUDIT LOG"):
            st.code("\n".join(reversed(ss.audit_log[-30:])) or "- no events -", language="text")


# ======================================================================
# Main
# ======================================================================
def init_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [], "tool_queue": [], "awaiting": None, "attachments": [],
        "audit_log": [], "quick_prompt": None, "last_error": None, "can_retry": False,
        "uploader_nonce": 0, "flash": None, "key_status": None, "user_title": "Sir",
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)
    ss.setdefault("groq_api_key", os.environ.get("GROQ_API_KEY", ""))


def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    if ss.flash:
        st.toast(ss.flash)
        ss.flash = None

    render_sidebar()
    render_header()
    render_history()
    render_gate()

    if ss.last_error:
        st.error(ss.last_error)
        if ss.can_retry and st.button("↻ RETRY LAST REQUEST", key="btn_retry"):
            with st.spinner("J.A.R.V.I.S. is retrying..."):
                agent_loop()
            st.rerun()

    locked = ss.awaiting is not None
    has_key = bool(get_api_key())
    if locked:
        placeholder = "Authorization pending - approve or deny the action above."
    elif not has_key:
        placeholder = "Enter your Groq API key in the sidebar to activate J.A.R.V.I.S."
    else:
        placeholder = f"At your service, {get_title()}. Type a command..."

    typed = st.chat_input(placeholder, disabled=locked or not has_key)
    prompt = typed or ss.quick_prompt
    ss.quick_prompt = None

    if prompt and has_key and not locked:
        ss.messages.append({"role": "user", "content": compose_user_content(prompt)})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        with st.spinner("J.A.R.V.I.S. is processing..."):
            agent_loop()
        st.rerun()


main()
