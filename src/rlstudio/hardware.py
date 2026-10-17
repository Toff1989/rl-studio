"""Detection du materiel et des logiciels installes.

Bibliotheque standard uniquement : ce module est execute par l'installateur (avant que le
runtime de calcul existe) et par la fenetre Parametres. Aucune fonction ne doit lever d'exception.
"""
import ctypes
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # pas de console qui clignote


def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW,
                           encoding="utf-8", errors="replace")
        return r.returncode, r.stdout
    except Exception:
        return -1, ""


def _powershell_json(script):
    rc, out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=30)
    try:
        data = json.loads(out)
    except ValueError:
        return []
    return data if isinstance(data, list) else [data]


# -- materiel -----------------------------------------------------------------------------
def scan_cpu() -> dict:
    info = {"name": platform.processor() or "inconnu", "cores": None, "threads": os.cpu_count()}
    if sys.platform == "win32":
        rows = _powershell_json("Get-CimInstance Win32_Processor | "
                                "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress")
        if rows:
            info["name"] = str(rows[0].get("Name", info["name"])).strip()
            info["cores"] = sum(int(r.get("NumberOfCores") or 0) for r in rows) or None
            info["threads"] = sum(int(r.get("NumberOfLogicalProcessors") or 0) for r in rows) or info["threads"]
    return info


def scan_ram_gb():
    if sys.platform != "win32":
        return None
    try:
        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MemStatus()
        st.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return round(st.ullTotalPhys / 1024 ** 3, 1)
    except Exception:
        return None


def _nvidia_smi():
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for cand in (r"C:\Windows\System32\nvidia-smi.exe", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if os.path.exists(cand):
            return cand
    return None


def parse_nvidia_smi(csv_text: str, plain_text: str = "") -> tuple[list, str | None]:
    """-> (liste de GPU NVIDIA, version CUDA maximale geree par le pilote)."""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        cc = parts[3] if len(parts) > 3 and re.fullmatch(r"\d+\.\d+", parts[3]) else None
        try:
            mem_gb = round(float(parts[2]) / 1024, 1)
        except ValueError:
            mem_gb = None
        gpus.append({"vendor": "NVIDIA", "name": parts[0], "driver": parts[1], "memory_gb": mem_gb,
                     "compute_cap": cc})
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", plain_text)
    return gpus, (m.group(1) if m else None)


def scan_gpus() -> tuple[list, str | None]:
    gpus, cuda = [], None
    smi = _nvidia_smi()
    if smi:
        rc, out = _run([smi, "--query-gpu=name,driver_version,memory.total,compute_cap",
                        "--format=csv,noheader,nounits"])
        if rc != 0:  # anciens pilotes : pas de compute_cap
            rc, out = _run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
        _, plain = _run([smi])
        gpus, cuda = parse_nvidia_smi(out if rc == 0 else "", plain)
    if sys.platform == "win32":  # autres cartes (AMD, Intel) : information seulement
        for row in _powershell_json("Get-CimInstance Win32_VideoController | "
                                    "Select-Object Name,DriverVersion | ConvertTo-Json -Compress"):
            name = str(row.get("Name", "")).strip()
            if name and "nvidia" not in name.lower() and "microsoft" not in name.lower():
                gpus.append({"vendor": "autre", "name": name, "driver": row.get("DriverVersion"),
                             "memory_gb": None, "compute_cap": None})
    return gpus, cuda


# -- MATLAB -------------------------------------------------------------------------------
def inspect_matlab(root) -> dict | None:
    """Decrit une installation MATLAB (None si le dossier n'en est pas une)."""
    root = Path(root)
    if not (root / "VersionInfo.xml").exists() and not (root / "bin").is_dir():
        return None
    release = version = None
    try:
        xml = (root / "VersionInfo.xml").read_text(encoding="utf-8", errors="replace")
        release = (re.search(r"<release>\s*(R\d{4}[ab])\s*</release>", xml) or [None, None])[1]
        version = (re.search(r"<version>\s*([\d.]+)\s*</version>", xml) or [None, None])[1]
    except OSError:
        pass
    engine_dir = root / "extern" / "engines" / "python"
    py_versions = []
    try:
        setup = (engine_dir / "setup.py").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"_supported_versions\s*=\s*\[(.*?)\]", setup, re.S)
        py_versions = re.findall(r"(\d+\.\d+)", m.group(1)) if m else []
    except OSError:
        pass
    has_exe = (root / "bin" / "matlab.exe").exists() or (root / "bin" / "matlab").exists()
    return {
        "root": str(root), "release": release, "version": version,
        "engine_series": ".".join(version.split(".")[:2]) if version else None,  # R2026b -> 26.2
        "simulink": (root / "toolbox" / "simulink").is_dir(),
        "engine_python_versions": py_versions,
        "complete": bool(has_exe and engine_dir.is_dir()),
    }


def _matlab_root_candidates() -> list[str]:
    roots = []
    if sys.platform == "win32":
        try:
            import winreg

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(hive, r"SOFTWARE\MathWorks\MATLAB", 0,
                                        winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
                        for i in range(winreg.QueryInfoKey(k)[0]):
                            with winreg.OpenKey(k, winreg.EnumKey(k, i)) as sub:
                                try:
                                    roots.append(winreg.QueryValueEx(sub, "MATLABROOT")[0])
                                except OSError:
                                    pass
                except OSError:
                    pass
        except ImportError:
            pass
        bases = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
        for letter in "CDEFGH":
            bases += [f"{letter}:\\", f"{letter}:\\Program Files"]
        for base in filter(None, bases):
            roots += glob.glob(os.path.join(base, "MATLAB", "R20*"))
    else:
        roots += glob.glob("/usr/local/MATLAB/R20*") + glob.glob(os.path.expanduser("~/MATLAB/R20*"))
    return roots


def scan_matlab() -> list[dict]:
    seen, found = set(), []
    for r in _matlab_root_candidates():
        try:
            key = str(Path(r).resolve()).lower()
        except OSError:
            continue
        if key in seen or not Path(r).is_dir():
            continue
        seen.add(key)
        info = inspect_matlab(r)
        if info:
            found.append(info)
    return sorted(found, key=lambda m: m.get("version") or "", reverse=True)


def scan_pythons() -> list[dict]:
    out = []
    if sys.platform == "win32":
        rc, text = _run(["py", "-0p"])
        for line in text.splitlines():
            m = re.match(r"\s*-V:(\S+)\s+\*?\s*(.+python[w]?\.exe)\s*$", line, re.I)
            if m:
                out.append({"version": m.group(1), "path": m.group(2).strip()})
    return out


def scan() -> dict:
    def safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    gpus, cuda = safe(scan_gpus, ([], None))
    return {
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": {"name": platform.platform(), "machine": platform.machine(), "python_bits": 64 if sys.maxsize > 2 ** 32 else 32},
        "cpu": safe(scan_cpu, {}),
        "ram_gb": safe(scan_ram_gb, None),
        "gpus": gpus,
        "cuda_driver": cuda,
        "matlab": safe(scan_matlab, []),
        "pythons": safe(scan_pythons, []),
    }


def summary(hw: dict) -> str:
    """Resume lisible (affiche par l'installateur et la fenetre Parametres)."""
    lines = []
    cpu = hw.get("cpu") or {}
    lines.append(f"Processeur : {cpu.get('name', 'inconnu')} ({cpu.get('threads', '?')} threads)")
    if hw.get("ram_gb"):
        lines.append(f"Memoire : {hw['ram_gb']} Go")
    gpus = hw.get("gpus") or []
    if not gpus:
        lines.append("Carte graphique : aucune detectee")
    for g in gpus:
        extra = []
        if g.get("memory_gb"):
            extra.append(f"{g['memory_gb']} Go")
        if g.get("driver"):
            extra.append(f"pilote {g['driver']}")
        lines.append(f"Carte graphique : {g['name']}" + (f" ({', '.join(extra)})" if extra else ""))
    if any(g.get("vendor") == "NVIDIA" for g in gpus):
        lines.append(f"CUDA maximal du pilote : {hw.get('cuda_driver') or 'inconnu'}")
    ml = hw.get("matlab") or []
    if not ml:
        lines.append("MATLAB : non detecte")
    for m in ml:
        state = "" if m.get("complete") else " - installation INCOMPLETE"
        lines.append(f"MATLAB {m.get('release') or '?'} : {m['root']} "
                     f"(Simulink {'oui' if m.get('simulink') else 'non'}){state}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(scan()))
