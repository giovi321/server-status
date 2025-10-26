#!/usr/bin/env python3
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import yaml  # PyYAML
except Exception:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml paho-mqtt", file=sys.stderr)
    raise

try:
    import paho.mqtt.client as mqtt
except Exception:
    print("Missing dependency: paho-mqtt. Install with: pip install paho-mqtt", file=sys.stderr)
    raise

# -----------------------------
# Data classes
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
    disks: bool = True      # mount usage
    raids: bool = True
    health: bool = True     # HDSentinel
    gpu: bool = True        # NVIDIA GPU via nvidia-smi

@dataclass
class Config:
    mqtt: MQTTConfig
    device: DeviceConfig
    modules: ModulesConfig
    mounts: Optional[Dict[str, str]]
    disks: Optional[List[str]]
    raids: Optional[List[str]]
    hdsentinel_path: str = "/root/HDSentinel"
    hdsentinel_min_interval_seconds: int = 1800
    hdsentinel_timeout_seconds: int = 60
    hdsentinel_cache_path: str = "/var/tmp/server_status_hdsentinel.json"
    cpu_temp_label: Optional[str] = None
    availability_topic: Optional[str] = None
    loop_seconds: Optional[int] = None

# -----------------------------
# Config
# -----------------------------
def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    mqtt_cfg = MQTTConfig(**raw["mqtt"])
    dev_cfg = DeviceConfig(**raw["device"])

    modules_raw = raw.get("modules", {})
    modules_cfg = ModulesConfig(
        cpu_usage=modules_raw.get("cpu_usage", True),
        cpu_temp=modules_raw.get("cpu_temp", True),
        memory=modules_raw.get("memory", True),
        uptime=modules_raw.get("uptime", True),
        disks=modules_raw.get("disks", True),
        raids=modules_raw.get("raids", True),
        health=modules_raw.get("health", True),
        gpu=modules_raw.get("gpu", True),
    )

    mounts = raw.get("mounts")
    disks = raw.get("disks")
    raids = raw.get("raids")

    return Config(
        mqtt=mqtt_cfg,
        device=dev_cfg,
        modules=modules_cfg,
        mounts=mounts,
        disks=disks,
        raids=raids,
        hdsentinel_path=raw.get("hdsentinel_path", "/root/HDSentinel"),
        hdsentinel_min_interval_seconds=raw.get("hdsentinel_min_interval_seconds", 1800),
        hdsentinel_timeout_seconds=raw.get("hdsentinel_timeout_seconds", 60),
        hdsentinel_cache_path=raw.get("hdsentinel_cache_path", "/var/tmp/server_status_hdsentinel.json"),
        cpu_temp_label=raw.get("cpu_temp_label"),
        availability_topic=raw.get("availability_topic"),
        loop_seconds=raw.get("loop_seconds"),
    )

# -----------------------------
# Helpers
# -----------------------------
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
    if usage < 0:
        usage = 0.0
    if usage > 100.0:
        usage = 100.0
    return usage

def read_cpu_temp_w_sensors(preferred_label: Optional[str]) -> Optional[float]:
    out = run(["/usr/bin/sensors"])
    if not out:
        return None
    lines = out.splitlines()
    if preferred_label:
        key = preferred_label.rstrip(":")
        for ln in lines:
            if ln.strip().startswith(key + ":"):
                for tok in ln.split():
                    if any(c.isdigit() for c in tok):
                        num = "".join(ch for ch in tok if ch.isdigit() or ch == ".")
                        if num:
                            try:
                                return float(num)
                            except Exception:
                                pass
    import re
    rx = re.compile(r"([0-9]+(\.[0-9]+)?)\s*°?C")
    for ln in lines:
        m = rx.search(ln)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None

def _mount_source_to_target_map() -> Dict[str, str]:
    m = {}
    try:
        with open("/proc/self/mounts") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) >= 2:
                    src = parts[0]
                    tgt = parts[1]
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

def disk_usage_percent(path: str) -> Optional[float]:
    try:
        mp = resolve_path_to_mountpoint(path)
        st = os.statvfs(mp)
        total = st.f_blocks * st.f_frsize
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
        if total <= 0:
            return None
        return used * 100.0 / total
    except Exception:
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
        total = mem.get("MemTotal")
        avail = mem.get("MemAvailable")
        if total and avail:
            return avail * 100.0 / total
    except Exception:
        pass
    return None

def uptime_days() -> Optional[float]:
    try:
        with open("/proc/uptime") as f:
            s = f.read().split()[0]
            seconds = float(s)
            return seconds / 86400.0
    except Exception:
        return None


# -----------------------------
# NVIDIA GPU via nvidia-smi
# -----------------------------
def read_nvidia_metrics(timeout: int = 3) -> Optional[dict]:
    """Return dict with temp_c, util_pct, mem_avail_pct using nvidia-smi.
    Requires NVIDIA drivers and nvidia-smi present. Returns None if unavailable.
    """
    smi = "/usr/bin/nvidia-smi"
    if not os.path.isfile(smi):
        # fallback: search in PATH
        for p in os.getenv("PATH", "").split(os.pathsep):
            cand = os.path.join(p, "nvidia-smi")
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                smi = cand
                break
        else:
            return None
    # Query: temperature.gpu, utilization.gpu, memory.total, memory.free
    out = run([smi, "--query-gpu=temperature.gpu,utilization.gpu,memory.total,memory.free", "--format=csv,noheader,nounits"], timeout=timeout)
    if not out:
        return None
    # Support multiple GPUs: take the first line
    line = out.strip().splitlines()[0].strip()
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        temp_c = float(parts[0])
        util_pct = float(parts[1])
        mem_total = float(parts[2])  # MiB
        mem_free = float(parts[3])   # MiB
        if mem_total <= 0:
            mem_avail_pct = None
        else:
            mem_avail_pct = mem_free * 100.0 / mem_total
        return {
            "temp_c": temp_c,
            "util_pct": util_pct,
            "mem_avail_pct": mem_avail_pct,
        }
    except Exception:
        return None

# -----------------------------
# HDSentinel parsing and throttling
# -----------------------------
def parse_hdsentinel(output: str, disks: List[str]) -> Dict[str, Optional[int]]:
    result = {d: None for d in disks}
    if not output:
        return result
    import re as _re
    rx = _re.compile(r"(\d{1,3})\s*%")
    for d in disks:
        for ln in output.splitlines():
            if d in ln.split():
                m = rx.search(ln)
                if m:
                    val = int(m.group(1))
                    if val < 0:
                        val = 0
                    if val > 100:
                        val = 100
                    result[d] = val
                    break
    return result

def _read_hdsentinel_cache(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _write_hdsentinel_cache(path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass

def hdsentinel_health(hdsentinel_path: str,
                      disks: List[str],
                      min_interval: int,
                      timeout_s: int,
                      cache_path: str) -> Dict[str, Optional[int]]:
    now = int(time.time())
    cached = _read_hdsentinel_cache(cache_path)
    if cached and isinstance(cached, dict):
        last = int(cached.get("ts", 0))
        if now - last < int(min_interval):
            cached_vals = cached.get("values", {})
            return {d: cached_vals.get(d) for d in disks}

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

    _write_hdsentinel_cache(cache_path, {"ts": now, "values": vals})
    return vals

# -----------------------------
# mdadm
# -----------------------------
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

# -----------------------------
# MQTT helpers
# -----------------------------
def ha_sensor_config(sensor_id: str,
                     name: str,
                     state_topic: str,
                     unit: Optional[str],
                     device_class: Optional[str],
                     mqtt_cfg: MQTTConfig,
                     device: DeviceConfig,
                     availability_topic: Optional[str]):
    payload = {
        "name": name,
        "state_topic": state_topic,
        "unique_id": sensor_id,
        "qos": mqtt_cfg.qos,
        "retain": mqtt_cfg.retain,
        "device": {
            "identifiers": device.identifiers,
            "name": device.name,
        },
        "state_class": "measurement",
    }
    if unit:
        payload["unit_of_measurement"] = unit
    if device_class:
        payload["device_class"] = device_class
    if availability_topic:
        payload["availability_topic"] = availability_topic
    if device.manufacturer:
        payload["device"]["manufacturer"] = device.manufacturer
    if device.model:
        payload["device"]["model"] = device.model
    if device.sw_version:
        payload["device"]["sw_version"] = device.sw_version
    return payload

def connect_mqtt(cfg: MQTTConfig) -> mqtt.Client:
    cid = cfg.client_id or f"server-status-{socket.gethostname()}"
    try:
        # Paho ≥2.0: request new callback API explicitly
        client = mqtt.Client(
            client_id=cid,
            userdata=None,
            protocol=mqtt.MQTTv311,  # or mqtt.MQTTv5
            transport="tcp",
            callback_api_version=mqtt.CallbackAPIVersion.V5
        )
    except AttributeError:
        # Older Paho without CallbackAPIVersion enum
        client = mqtt.Client(client_id=cid, userdata=None, protocol=mqtt.MQTTv311, transport="tcp")

    if cfg.username:
        client.username_pw_set(cfg.username, cfg.password or "")
    if cfg.tls:
        client.tls_set(ca_certs=cfg.ca_certs)
        if cfg.insecure_tls:
            client.tls_insecure_set(True)
    client.connect(cfg.host, cfg.port, cfg.keepalive)
    return client

def publish(client: mqtt.Client, topic: str, payload: str, cfg: MQTTConfig):
    client.publish(topic, payload=payload, qos=cfg.qos, retain=cfg.retain)

def publish_availability(client: mqtt.Client, avail_topic: str, online: bool, cfg: MQTTConfig):
    publish(client, avail_topic, "online" if online else "offline", cfg)

def clamp_round(x: Optional[float], ndigits: int = 0) -> Optional[float]:
    if x is None:
        return None
    return round(x, ndigits)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Publish server status to MQTT with optional Home Assistant discovery.")
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    ap.add_argument("--once", action="store_true", help="Run once and exit even if loop_seconds is set.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mqtt_cfg = cfg.mqtt
    base = mqtt_cfg.base_topic.rstrip("/")
    avail_topic = cfg.availability_topic or f"{base}/availability"

    client = connect_mqtt(mqtt_cfg)
    client.will_set(avail_topic, "offline", qos=mqtt_cfg.qos, retain=True)
    client.loop_start()
    publish_availability(client, avail_topic, True, mqtt_cfg)

    # Discovery
    if mqtt_cfg.discovery_enable:

        disc = mqtt_cfg.discovery_prefix.rstrip("/")
        node = base.replace("/", "_")

        # CPU usage
        if cfg.modules.cpu_usage:
            cpu_id = f"{node}_cpu_usage"
            cpu_state = f"{base}/cpu_usage"
            cpu_cfg = ha_sensor_config(cpu_id, "CPU Usage", cpu_state, "%", None, mqtt_cfg, cfg.device, avail_topic)
            client.publish(f"{disc}/sensor/{node}/{cpu_id}/config", json.dumps(cpu_cfg), retain=True, qos=mqtt_cfg.qos)

        # CPU temp
        if cfg.modules.cpu_temp:
            cput_id = f"{node}_cpu_temp"
            cput_state = f"{base}/cpu_temp"
            cput_cfg = ha_sensor_config(cput_id, "CPU Temp", cput_state, "°C", "temperature", mqtt_cfg, cfg.device, avail_topic)
            client.publish(f"{disc}/sensor/{node}/{cput_id}/config", json.dumps(cput_cfg), retain=True, qos=mqtt_cfg.qos)

        # Memory available
        if cfg.modules.memory:
            mem_id = f"{node}_memory_available"
            mem_state = f"{base}/memory_available"
            mem_cfg = ha_sensor_config(mem_id, "Memory Available", mem_state, "%", None, mqtt_cfg, cfg.device, avail_topic)
            client.publish(f"{disc}/sensor/{node}/{mem_id}/config", json.dumps(mem_cfg), retain=True, qos=mqtt_cfg.qos)

        # Uptime
        if cfg.modules.uptime:
            up_id = f"{node}_uptime_days"
            up_state = f"{base}/uptime_days"
            up_cfg = ha_sensor_config(up_id, "Uptime", up_state, "d", None, mqtt_cfg, cfg.device, avail_topic)
            client.publish(f"{disc}/sensor/{node}/{up_id}/config", json.dumps(up_cfg), retain=True, qos=mqtt_cfg.qos)

        # Mounts
        if cfg.modules.disks and cfg.mounts:
            for key in cfg.mounts.keys():
                sid = f"{node}_disk_usage_{key}"
                st = f"{base}/disk_usage/{key}"
                sc = ha_sensor_config(sid, f"Disk Usage {key}", st, "%", None, mqtt_cfg, cfg.device, avail_topic)
                client.publish(f"{disc}/sensor/{node}/{sid}/config", json.dumps(sc), retain=True, qos=mqtt_cfg.qos)

        # Disks health
        if cfg.modules.health and cfg.disks:
            for d in cfg.disks:
                sid = f"{node}_health_{d}"
                st = f"{base}/health_{d}"
                sc = ha_sensor_config(sid, f"Health {d}", st, "%", None, mqtt_cfg, cfg.device, avail_topic)
                client.publish(f"{disc}/sensor/{node}/{sid}/config", json.dumps(sc), retain=True, qos=mqtt_cfg.qos)

        # GPU (NVIDIA)
        if cfg.modules.gpu:
            gpu_temp_id = f"{node}_gpu_temp"
            gpu_temp_state = f"{base}/gpu/temp"
            gpu_temp_cfg = ha_sensor_config(gpu_temp_id, "GPU Temp", gpu_temp_state, "°C", "temperature", mqtt_cfg, cfg.device, avail_topic)
            client.publish(f"{disc}/sensor/{node}/{gpu_temp_id}/config", json.dumps(gpu_temp_cfg), retain=True, qos=mqtt_cfg.qos)

            gpu_util_id = f"{node}_gpu_util"
            gpu_util_state = f"{base}/gpu/util"
            gpu_util_cfg = ha_sensor_config(gpu_util_id, "GPU Utilization", gpu_util_state, "%", None, mqtt_cfg, cfg.device, avail_topic)
            client.publish(f"{disc}/sensor/{node}/{gpu_util_id}/config", json.dumps(gpu_util_cfg), retain=True, qos=mqtt_cfg.qos)

            gpu_mem_id = f"{node}_gpu_mem_available"
            gpu_mem_state = f"{base}/gpu/mem_available"
            gpu_mem_cfg = ha_sensor_config(gpu_mem_id, "GPU Memory Available", gpu_mem_state, "%", None, mqtt_cfg, cfg.device, avail_topic)
            client.publish(f"{disc}/sensor/{node}/{gpu_mem_id}/config", json.dumps(gpu_mem_cfg), retain=True, qos=mqtt_cfg.qos)


        # Raids
        if cfg.modules.raids and cfg.raids:
            for arr in cfg.raids:
                sid = f"{node}_raid_{arr}"
                st = f"{base}/raid/{arr}"
                sc = ha_sensor_config(sid, f"RAID {arr} Active", st, None, None, mqtt_cfg, cfg.device, avail_topic)
                client.publish(f"{disc}/sensor/{node}/{sid}/config", json.dumps(sc), retain=True, qos=mqtt_cfg.qos)

    def one_cycle():
        # NVIDIA GPU
        if cfg.modules.gpu:
            gm = read_nvidia_metrics()
            if gm:
                if gm.get("temp_c") is not None:
                    client.publish(f"{base}/gpu/temp", payload=f"{int(round(gm['temp_c']))}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)
                if gm.get("util_pct") is not None:
                    client.publish(f"{base}/gpu/util", payload=f"{int(round(gm['util_pct']))}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)
                if gm.get("mem_avail_pct") is not None:
                    client.publish(f"{base}/gpu/mem_available", payload=f"{int(round(gm['mem_avail_pct']))}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)

        # CPU usage
        if cfg.modules.cpu_usage:
            cpu = clamp_round(read_cpu_usage_one_second(), 0)
            if cpu is not None:
                client.publish(f"{base}/cpu_usage", payload=f"{int(cpu)}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)

        # CPU temp (not a percentage; one decimal)
        if cfg.modules.cpu_temp:
            ctemp = read_cpu_temp_w_sensors(cfg.cpu_temp_label)
            if ctemp is not None:
                client.publish(f"{base}/cpu_temp", payload=f"{ctemp:.1f}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)

        # Memory available
        if cfg.modules.memory:
            mem = memory_available_percent()
            if mem is not None:
                client.publish(f"{base}/memory_available", payload=f"{int(round(mem))}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)

        # Uptime
        if cfg.modules.uptime:
            up = uptime_days()
            if up is not None:
                if up < 10:
                    client.publish(f"{base}/uptime_days", payload=f"{up:.2f}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)
                else:
                    client.publish(f"{base}/uptime_days", payload=f"{int(round(up))}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)

        # Mounts usage
        if cfg.modules.disks and cfg.mounts:
            for key, path in cfg.mounts.items():
                pct = disk_usage_percent(path)
                if pct is not None:
                    client.publish(f"{base}/disk_usage/{key}", payload=f"{int(round(pct))}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)

        # HDSentinel
        if cfg.modules.health and cfg.disks:
            hs = hdsentinel_health(cfg.hdsentinel_path,
                                   cfg.disks,
                                   cfg.hdsentinel_min_interval_seconds,
                                   cfg.hdsentinel_timeout_seconds,
                                   cfg.hdsentinel_cache_path)
            for d, val in hs.items():
                if val is not None:
                    client.publish(f"{base}/health_{d}", payload=f"{int(val)}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)

        # mdadm
        if cfg.modules.raids and cfg.raids:
            for arr in cfg.raids:
                val = mdadm_active_devices(arr)
                if val is not None:
                    client.publish(f"{base}/raid/{arr}", payload=f"{int(val)}", qos=mqtt_cfg.qos, retain=mqtt_cfg.retain)

    try:
        if cfg.loop_seconds and not args.once:
            interval = max(5, int(cfg.loop_seconds))
            while True:
                one_cycle()
                time.sleep(interval)
        else:
            one_cycle()
    finally:
        publish_availability(client, avail_topic, False, mqtt_cfg)
        time.sleep(0.1)
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
