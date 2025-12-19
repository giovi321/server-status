#!/usr/bin/env python3
# server-status.py

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import yaml
except Exception:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    raise

try:
    import paho.mqtt.client as mqtt
    try:
        from paho.mqtt.client import CallbackAPIVersion  # paho >=2
    except Exception:
        CallbackAPIVersion = None
except Exception:
    print("Missing dependency: paho-mqtt. Install with: pip install paho-mqtt", file=sys.stderr)
    raise

# -----------------------------
# Config structures
# -----------------------------
@dataclass
class MQTTConfig:
    host: str
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: Optional[str] = None
    base_topic: str = "SERVER"
    retain: bool = False
    qos: int = 0
    tls: bool = False
    ca_certs: Optional[str] = None
    insecure_tls: bool = False
    keepalive: int = 30
    discovery_enable: bool = True
    discovery_prefix: str = "homeassistant"

@dataclass
class DeviceConfig:
    name: str
    identifiers: List[str]
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    sw_version: Optional[str] = None

@dataclass
class ModulesConfig:
    cpu_usage: bool = True
    cpu_temp: bool = True
    memory: bool = True
    uptime: bool = True
    disks: bool = True
    raids: bool = True
    health: bool = True
    gpu: bool = True
    apt_updates: bool = True
    docker_updates: bool = True

@dataclass
class Config:
    mqtt: MQTTConfig
    device: DeviceConfig
    modules: ModulesConfig
    mounts: Optional[Dict[str, str]] = None
    disks: Optional[List[str]] = None
    raids: Optional[List[str]] = None
    hdsentinel_path: str = "/root/HDSentinel"
    hdsentinel_min_interval_seconds: int = 1800
    hdsentinel_timeout_seconds: int = 60
    hdsentinel_cache_path: str = "/var/tmp/server_status_hdsentinel.json"
    apt_min_interval_seconds: int = 3600
    apt_cache_path: str = "/var/tmp/server_status_apt.json"
    docker_min_interval_seconds: int = 21600
    docker_cache_path: str = "/var/tmp/server_status_docker.json"
    cpu_temp_label: Optional[str] = None
    availability_topic: Optional[str] = None
    loop_seconds: Optional[int] = 60

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    mqtt_cfg = MQTTConfig(**raw["mqtt"])
    dev_cfg = DeviceConfig(**raw["device"])
    mraw = raw.get("modules", {})
    modules = ModulesConfig(
        cpu_usage=mraw.get("cpu_usage", True),
        cpu_temp=mraw.get("cpu_temp", True),
        memory=mraw.get("memory", True),
        uptime=mraw.get("uptime", True),
        disks=mraw.get("disks", True),
        raids=mraw.get("raids", True),
        health=mraw.get("health", True),
        gpu=mraw.get("gpu", True),
        apt_updates=mraw.get("apt_updates", True),
        docker_updates=mraw.get("docker_updates", True),
    )
    return Config(
        mqtt=mqtt_cfg,
        device=dev_cfg,
        modules=modules,
        mounts=raw.get("mounts"),
        disks=raw.get("disks"),
        raids=raw.get("raids"),
        hdsentinel_path=raw.get("hdsentinel_path", "/root/HDSentinel"),
        hdsentinel_min_interval_seconds=raw.get("hdsentinel_min_interval_seconds", 1800),
        hdsentinel_timeout_seconds=raw.get("hdsentinel_timeout_seconds", 60),
        hdsentinel_cache_path=raw.get("hdsentinel_cache_path", "/var/tmp/server_status_hdsentinel.json"),
        apt_min_interval_seconds=raw.get("apt_min_interval_seconds", 3600),
        apt_cache_path=raw.get("apt_cache_path", "/var/tmp/server_status_apt.json"),
        docker_min_interval_seconds=raw.get("docker_min_interval_seconds", 21600),
        docker_cache_path=raw.get("docker_cache_path", "/var/tmp/server_status_docker.json"),
        cpu_temp_label=raw.get("cpu_temp_label"),
        availability_topic=raw.get("availability_topic"),
        loop_seconds=raw.get("loop_seconds", 60),
    )

def run(cmd: List[str], timeout: int = 5) -> str:
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False, text=True)
        return out.stdout
    except Exception:
        return ""

def read_cpu_usage_one_second() -> float:
    def read_stat():
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = line.split()
                    vals = list(map(int, parts[1:8]))
                    user, nice, system, idle, iowait, irq, softirq = vals[:7]
                    idle_all = idle + iowait
                    non_idle = user + nice + system + irq + softirq
                    total = idle_all + non_idle
                    return non_idle, total
        return 0, 1
    n1, t1 = read_stat()
    time.sleep(1.0)
    n2, t2 = read_stat()
    delta = max(1, t2 - t1)
    usage = (n2 - n1) * 100.0 / delta
    return max(0.0, min(100.0, usage))

def read_cpu_temp_w_sensors(preferred_label: Optional[str]) -> Optional[float]:
    out = run(["/usr/bin/sensors"])
    if not out:
        return None
    import re
    lines = out.splitlines()

    def temp_in(s: str) -> Optional[float]:
        m = re.search(r'([-+]?\d+(?:\.\d+)?)\s*°?C', s)
        return float(m.group(1)) if m else None

    # 1) Exact label, but match anywhere left of the first colon (robust to spacing/case)
    if preferred_label:
        key = preferred_label.strip().rstrip(":").lower()
        for ln in lines:
            if ":" not in ln:
                continue
            left, right = ln.split(":", 1)
            if key in left.strip().lower():
                v = temp_in(right)
                if v is not None:
                    return v

    # 2) Fallbacks
    for lab in ["Tctl", "Tdie", "Package id 0", "Composite", "CPU"]:
        k = lab.lower()
        for ln in lines:
            if ":" not in ln:
                continue
            left, right = ln.split(":", 1)
            if k in left.strip().lower():
                v = temp_in(right)
                if v is not None:
                    return v

    # 3) Last resort: first temperature anywhere
    for ln in lines:
        v = temp_in(ln)
        if v is not None:
            return v
    return None

def _mount_source_to_target_map() -> Dict[str, str]:
    m = {}
    try:
        with open("/proc/self/mounts") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) >= 2:
                    src = parts[0]; tgt = parts[1]
                    m[src] = tgt
    except Exception:
        pass
    return m

_MOUNT_MAP = None
def resolve_path_to_mountpoint(path: str) -> str:
    global _MOUNT_MAP
    if os.path.isdir(path):
        return path
    if _MOUNT_MAP is None:
        _MOUNT_MAP = _mount_source_to_target_map()
    if path in _MOUNT_MAP:
        return _MOUNT_MAP[path]
    try:
        real = os.path.realpath(path)
        if real in _MOUNT_MAP:
            return _MOUNT_MAP[real]
    except Exception:
        pass
    return path

def is_mount_point_accessible(path: str) -> bool:
    """Check if a mount point is accessible and readable."""
    try:
        return os.path.isdir(path) and os.access(path, os.R_OK)
    except Exception:
        return False

def disk_usage_percent(path: str) -> Optional[float]:
    try:
        mp = resolve_path_to_mountpoint(path)
        if not is_mount_point_accessible(mp):
            print(f"Mount point {mp} (path: {path}) is not accessible", file=sys.stderr)
            return None
        st = os.statvfs(mp)
        total = st.f_blocks * st.f_frsize
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
        if total <= 0:
            print(f"Warning: Invalid total size ({total}) for mount point {mp} (path: {path})", file=sys.stderr)
            return None
        usage = used * 100.0 / total
        return usage
    except OSError as e:
        print(f"OS error getting disk usage for {path} (mount: {mp if 'mp' in locals() else 'unknown'}): {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error getting disk usage for {path}: {e}", file=sys.stderr)
        return None

def memory_available_percent() -> Optional[float]:
    mem = {}
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                parts = ln.split(":")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    if val.isdigit():
                        mem[key] = int(val)
        total = mem.get("MemTotal"); avail = mem.get("MemAvailable")
        if total and avail:
            return avail * 100.0 / total
    except Exception:
        pass
    return None

def uptime_days() -> Optional[float]:
    try:
        with open("/proc/uptime") as f:
            s = f.read().split()[0]
            return float(s) / 86400.0
    except Exception:
        return None

def parse_hdsentinel(output: str, disks: List[str]) -> Dict[str, Optional[int]]:
    # Health is the second numeric token after the device token.
    import os, re
    res = {d: None for d in disks}
    if not output:
        return res
    num_rx = re.compile(r"\\d+")
    aliases = {}
    for d in disks:
        base = os.path.basename(d)
        aliases[d] = {d, base, f"/dev/{base}"}
    for ln in output.splitlines():
        toks = ln.split()
        if not toks:
            continue
        line_ids = {os.path.basename(t) if "/" in t else t for t in toks}
        line_ids |= {f"/dev/{t}" for t in line_ids}
        matched = None
        for d in disks:
            if aliases[d] & line_ids:
                matched = d
                break
        if not matched or res[matched] is not None:
            continue
        dev_idx = None
        ali = aliases[matched]
        for i, t in enumerate(toks):
            tb = os.path.basename(t) if "/" in t else t
            if t in ali or tb in ali or f"/dev/{tb}" in ali:
                dev_idx = i; break
        if dev_idx is None:
            continue
        tail = " ".join(toks[dev_idx+1:])
        nums = [int(m.group()) for m in num_rx.finditer(tail)]
        if not nums:
            continue
        health = nums[1] if len(nums) >= 2 else nums[0]
        health = max(0, min(100, int(health)))
        res[matched] = health
    return res

def _read_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json(path: str, obj: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except Exception:
        pass

def hdsentinel_health(hdsentinel_path: str,
                      disks: List[str],
                      min_interval: int,
                      timeout_s: int,
                      cache_path: str) -> Dict[str, Optional[int]]:
    now = int(time.time())
    cached = _read_json(cache_path)
    if cached and now - int(cached.get("ts", 0)) < int(min_interval):
        vals = cached.get("values", {})
        return {d: vals.get(d) for d in disks}
    if not disks or not os.path.isfile(hdsentinel_path):
        vals = {d: None for d in disks}
    else:
        out = run([hdsentinel_path, "-solid"], timeout=timeout_s)
        vals = parse_hdsentinel(out, disks)
        if any(vals[d] is None for d in disks):
            time.sleep(min(5, timeout_s // 4 if timeout_s else 2))
            out2 = run([hdsentinel_path, "-solid"], timeout=timeout_s)
            v2 = parse_hdsentinel(out2, disks)
            for d in disks:
                if vals[d] is None and v2.get(d) is not None:
                    vals[d] = v2[d]
    _write_json(cache_path, {"ts": now, "values": vals})
    return vals

def read_nvidia_metrics(timeout: int = 3) -> Optional[dict]:
    smi = "/usr/bin/nvidia-smi"
    if not os.path.isfile(smi):
        for p in os.getenv("PATH", "").split(os.pathsep):
            cand = os.path.join(p, "nvidia-smi")
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                smi = cand; break
        else:
            return None
    out = run([smi, "--query-gpu=temperature.gpu,utilization.gpu,memory.total,memory.free", "--format=csv,noheader,nounits"], timeout=timeout)
    if not out:
        return None
    line = out.strip().splitlines()[0].strip()
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        temp_c = float(parts[0]); util = float(parts[1])
        total = float(parts[2]); free = float(parts[3])
        mem_avail = None if total <= 0 else free * 100.0 / total
        return {"temp_c": temp_c, "util_pct": util, "mem_avail_pct": mem_avail}
    except Exception:
        return None

def apt_updates_count(timeout: int = 30) -> Optional[int]:
    out = run(["/usr/bin/apt-get", "-s", "dist-upgrade"], timeout=timeout)
    cnt = 0
    if out:
        for ln in out.splitlines():
            if ln.startswith("Inst "):
                cnt += 1
        if cnt > 0:
            return cnt
    out2 = run(["/usr/bin/apt", "list", "--upgradeable"], timeout=timeout)
    if out2:
        lines = [l for l in out2.splitlines() if l and not l.startswith("Listing...")]
        return max(0, len(lines))
    return None

def cached_apt_updates(min_interval: int, cache_path: str) -> Optional[int]:
    now = int(time.time())
    cached = _read_json(cache_path)
    if cached and now - int(cached.get("ts", 0)) < int(min_interval):
        return cached.get("count")
    val = apt_updates_count()
    _write_json(cache_path, {"ts": now, "count": val})
    return val

def docker_updates_count(timeout: int = 120) -> Optional[int]:
    ps = run(["/usr/bin/docker", "ps", "--format", "{{.Image}}"], timeout=15)
    if not ps:
        return None
    images = sorted(set([l.strip() for l in ps.splitlines() if l.strip()]))
    updates = 0
    for img in images:
        out = run(["/usr/bin/docker", "pull", img], timeout=timeout)
        if not out:
            continue
        low = out.lower()
        if "downloaded newer image" in low or "status: downloaded newer image" in low:
            updates += 1
    return updates

def cached_docker_updates(min_interval: int, cache_path: str) -> Optional[int]:
    now = int(time.time())
    cached = _read_json(cache_path)
    if cached and now - int(cached.get("ts", 0)) < int(min_interval):
        return cached.get("count")
    val = docker_updates_count()
    _write_json(cache_path, {"ts": now, "count": val})
    return val

def ha_sensor_config(sensor_id: str, name: str, state_topic: str, unit: Optional[str],
                     device_class: Optional[str], mqtt_cfg: MQTTConfig, device: DeviceConfig,
                     availability_topic: Optional[str]):
    payload = {
        "name": name,
        "state_topic": state_topic,
        "unique_id": sensor_id,
        "qos": mqtt_cfg.qos,
        "retain": mqtt_cfg.retain,
        "device": {"identifiers": device.identifiers, "name": device.name},
        "state_class": "measurement",
    }
    if unit: payload["unit_of_measurement"] = unit
    if device_class: payload["device_class"] = device_class
    if availability_topic: payload["availability_topic"] = availability_topic
    if device.manufacturer: payload["device"]["manufacturer"] = device.manufacturer
    if device.model: payload["device"]["model"] = device.model
    if device.sw_version: payload["device"]["sw_version"] = device.sw_version
    return payload

def safe_publish(client, topic: str, payload: str, cfg: MQTTConfig):
    try:
        if not client.is_connected():
            return
        client.publish(topic, payload=payload, qos=cfg.qos, retain=cfg.retain)
    except Exception:
        pass

def _publish_discovery(client, cfg: Config, base: str, avail_topic: str):
    if not cfg.mqtt.discovery_enable:
        return
    disc = cfg.mqtt.discovery_prefix.rstrip("/")
    node = base.replace("/", "_")
    def ha(sensor_id, name, state_topic, unit=None, device_class=None):
        payload = ha_sensor_config(sensor_id, name, state_topic, unit, device_class,
                                   cfg.mqtt, cfg.device, avail_topic)
        safe_publish(client, f"{disc}/sensor/{node}/{sensor_id}/config", json.dumps(payload), cfg.mqtt)
    if cfg.modules.cpu_usage: ha(f"{node}_cpu_usage", "CPU Usage", f"{base}/cpu_usage", "%")
    if cfg.modules.cpu_temp:  ha(f"{node}_cpu_temp", "CPU Temp", f"{base}/cpu_temp", "°C", "temperature")
    if cfg.modules.memory:    ha(f"{node}_memory_available", "Memory Available", f"{base}/memory_available", "%")
    if cfg.modules.uptime:    ha(f"{node}_uptime_days", "Uptime", f"{base}/uptime_days", "d")
    if cfg.modules.disks and cfg.mounts:
        for key in cfg.mounts.keys():
            ha(f"{node}_disk_usage_{key}", f"Disk Usage {key}", f"{base}/disk_usage/{key}", "%")
    if cfg.modules.health and cfg.disks:
        for d in cfg.disks:
            ha(f"{node}_health_{d}", f"Health {d}", f"{base}/health_{d}", "%")
    if cfg.modules.gpu:
        ha(f"{node}_gpu_temp", "GPU Temp", f"{base}/gpu/temp", "°C", "temperature")
        ha(f"{node}_gpu_util", "GPU Utilization", f"{base}/gpu/util", "%")
        ha(f"{node}_gpu_mem_available", "GPU Memory Available", f"{base}/gpu/mem_available", "%")
    if cfg.modules.apt_updates:
        ha(f"{node}_apt_updates", "APT Updates Available", f"{base}/updates/apt")
    if cfg.modules.docker_updates:
        ha(f"{node}_docker_updates", "Docker Image Updates", f"{base}/updates/docker")

def connect_mqtt(cfg: Config, base: str, avail_topic: str):
    try:
        client = mqtt.Client(client_id=(cfg.mqtt.client_id or f"server-status-{socket.gethostname()}"),
                             userdata=None,
                             protocol=mqtt.MQTTv311,
                             transport="tcp",
                             callback_api_version=(CallbackAPIVersion.V5 if CallbackAPIVersion else None))
    except Exception:
        client = mqtt.Client(client_id=(cfg.mqtt.client_id or f"server-status-{socket.gethostname()}"),
                             userdata=None,
                             protocol=mqtt.MQTTv311,
                             transport="tcp")
    if cfg.mqtt.username:
        client.username_pw_set(cfg.mqtt.username, cfg.mqtt.password or "")
    if cfg.mqtt.tls:
        client.tls_set(ca_certs=cfg.mqtt.ca_certs)
        if cfg.mqtt.insecure_tls:
            client.tls_insecure_set(True)
    client.will_set(avail_topic, "offline", qos=cfg.mqtt.qos, retain=True)

    def on_connect(cl, userdata, flags, rc, properties=None):
        ok = (rc == 0) or (getattr(rc, "value", None) == 0)
        if ok:
            safe_publish(client, avail_topic, "online", cfg.mqtt)
            _publish_discovery(client, cfg, base, avail_topic)

    reconnect_state = {"in_progress": False}
    reconnect_lock = threading.Lock()

    def _attempt_reconnect():
        delay = 5
        while True:
            try:
                client.reconnect()
                return
            except Exception:
                try:
                    client.connect_async(cfg.mqtt.host, cfg.mqtt.port, cfg.mqtt.keepalive)
                except Exception:
                    pass
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def on_disconnect(cl, userdata, rc, properties=None):
        ok = (rc == 0) or (getattr(rc, "value", None) == 0)
        if ok:
            return
        with reconnect_lock:
            if reconnect_state["in_progress"]:
                return
            reconnect_state["in_progress"] = True

        def worker():
            try:
                _attempt_reconnect()
            finally:
                with reconnect_lock:
                    reconnect_state["in_progress"] = False

        threading.Thread(target=worker, daemon=True).start()

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    try:
        client.reconnect_delay_set(min_delay=5, max_delay=60)
    except Exception:
        pass
    client.connect_async(cfg.mqtt.host, cfg.mqtt.port, cfg.mqtt.keepalive)
    client.loop_start()
    return client

def mdadm_active_devices(arr: str) -> Optional[int]:
    out = run(["/usr/sbin/mdadm", "-D", f"/dev/{arr}"], timeout=5)
    if not out:
        return None
    for ln in out.splitlines():
        if "Active Devices" in ln:
            digits = "".join(ch for ch in ln if ch.isdigit())
            if digits.isdigit():
                return int(digits)
    return None

def main():
    ap = argparse.ArgumentParser(description="Publish server metrics to MQTT + Home Assistant discovery.")
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    ap.add_argument("--once", action="store_true", help="Run one cycle then exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = cfg.mqtt.base_topic.rstrip("/")
    avail_topic = cfg.availability_topic or f"{base}/availability"
    client = connect_mqtt(cfg, base, avail_topic)

    def one_cycle():
        if cfg.modules.cpu_usage:
            cpu = int(round(read_cpu_usage_one_second()))
            safe_publish(client, f"{base}/cpu_usage", f"{cpu}", cfg.mqtt)
        if cfg.modules.cpu_temp:
            ctemp = read_cpu_temp_w_sensors(cfg.cpu_temp_label)
            if ctemp is not None:
                safe_publish(client, f"{base}/cpu_temp", f"{ctemp:.1f}", cfg.mqtt)
        if cfg.modules.memory:
            mem = memory_available_percent()
            if mem is not None:
                safe_publish(client, f"{base}/memory_available", f"{int(round(mem))}", cfg.mqtt)
        if cfg.modules.uptime:
            up = uptime_days()
            if up is not None:
                if up < 10:
                    safe_publish(client, f"{base}/uptime_days", f"{up:.2f}", cfg.mqtt)
                else:
                    safe_publish(client, f"{base}/uptime_days", f"{int(round(up))}", cfg.mqtt)
        if cfg.modules.disks and cfg.mounts:
            for key, path in cfg.mounts.items():
                pct = disk_usage_percent(path)
                if pct is not None:
                    safe_publish(client, f"{base}/disk_usage/{key}", f"{int(round(pct))}", cfg.mqtt)
                else:
                    # Publish unknown status when disk usage cannot be determined
                    safe_publish(client, f"{base}/disk_usage/{key}", "unknown", cfg.mqtt)
        if cfg.modules.health and cfg.disks:
            hs = hdsentinel_health(cfg.hdsentinel_path, cfg.disks, cfg.hdsentinel_min_interval_seconds, cfg.hdsentinel_timeout_seconds, cfg.hdsentinel_cache_path)
            for d, val in hs.items():
                if val is not None:
                    safe_publish(client, f"{base}/health_{d}", f"{int(val)}", cfg.mqtt)
        if cfg.modules.gpu:
            gm = read_nvidia_metrics()
            if gm:
                if gm.get("temp_c") is not None:
                    safe_publish(client, f"{base}/gpu/temp", f"{int(round(gm['temp_c']))}", cfg.mqtt)
                if gm.get("util_pct") is not None:
                    safe_publish(client, f"{base}/gpu/util", f"{int(round(gm['util_pct']))}", cfg.mqtt)
                if gm.get("mem_avail_pct") is not None:
                    safe_publish(client, f"{base}/gpu/mem_available", f"{int(round(gm['mem_avail_pct']))}", cfg.mqtt)
        if cfg.modules.apt_updates:
            cnt = cached_apt_updates(cfg.apt_min_interval_seconds, cfg.apt_cache_path)
            if cnt is not None:
                safe_publish(client, f"{base}/updates/apt", f"{int(cnt)}", cfg.mqtt)
        if cfg.modules.docker_updates:
            cnt = cached_docker_updates(cfg.docker_min_interval_seconds, cfg.docker_cache_path)
            if cnt is not None:
                safe_publish(client, f"{base}/updates/docker", f"{int(cnt)}", cfg.mqtt)
        if cfg.modules.raids and cfg.raids:
            for arr in cfg.raids:
                val = mdadm_active_devices(arr)
                if val is not None:
                    safe_publish(client, f"{base}/raid/{arr}", f"{int(val)}", cfg.mqtt)

    try:
        if cfg.loop_seconds and not args.once:
            interval = max(5, int(cfg.loop_seconds))
            while True:
                one_cycle()
                time.sleep(interval)
        else:
            one_cycle()
    finally:
        try:
            safe_publish(client, avail_topic, "offline", cfg.mqtt)
            time.sleep(0.1)
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()
