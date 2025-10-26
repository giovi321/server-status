# Server Status MQTT Publisher

Publishes key server metrics (CPU, memory, uptime, disk usage, RAID, drive health, and optional NVIDIA GPU) to an MQTT broker with Home Assistant autodiscovery support.  
All configuration is YAML-based; each module can be toggled individually.

---

## Features

| Metric Type | Source | MQTT Topic Prefix | Notes |
|--------------|---------|-------------------|--------|
| CPU usage / temp | `/proc/stat`, `sensors` | `<base>/cpu_*` | Temp label customizable |
| Memory available % | `/proc/meminfo` | `<base>/memory_available` | Uses MemAvailable / MemTotal |
| Uptime (days) | `/proc/uptime` | `<base>/uptime_days` | 2 decimals if <10 days |
| Disk usage % | `statvfs` | `<base>/disk_usage/<mount>` | Automatically resolves device → mountpoint |
| RAID active members | `mdadm -D /dev/mdX` | `<base>/raid/<md>` | Only if listed |
| Drive health | `HDSentinel -solid` | `<base>/health_<disk>` | Cached, throttled (default 30 min) |
| NVIDIA GPU | `nvidia-smi` | `<base>/gpu/...` | Temp °C, Util %, VRAM % free |

---

## Requirements

- Debian ≥ 12
- Python ≥ 3.9  
- Packages: `paho-mqtt`, `PyYAML`  
- Optional tools:  
  - `lm-sensors` (CPU temp)  
  - `HDSentinel` CLI in `/root/HDSentinel`  
  - `mdadm` for RAID  
  - `nvidia-smi` for GPU metrics  

Install:
```bash
sudo apt install python3-venv lm-sensors mdadm
cd /root/server-status
python3 -m venv .
./bin/pip install paho-mqtt PyYAML
```

---

## Configuration

Edit `server-status.yaml`:

```yaml
mqtt:
  host: 192.168.1.65
  username: mqtt
  password: mqtt_password
  base_topic: SERVER

device:
  name: SERVER
  identifiers: ["SERVER"]

modules:
  cpu_usage: true
  cpu_temp: true
  memory: true
  uptime: true
  disks: true
  raids: true
  health: true
  gpu: true        # enable NVIDIA GPU metrics

mounts:
  root: /
  storage1: /dev/mapper/storage1
  backup: /dev/mapper/backup

disks: ["sda", "sdb"]
raids: ["md0", "md1"]

hdsentinel_path: /root/HDSentinel
loop_seconds: 60
```

Missing sections or `false` disable that metric entirely.

---

## Run manually

```bash
./bin/python3 server-status.py -c server-status.yaml
```

Run once and exit:
```bash
./bin/python3 server-status.py -c server-status.yaml --once
```

---

## Systemd Service

Create `/etc/systemd/system/server-status.service`:

```
[Unit]
Description=Server Status MQTT Publisher
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/server-status
ExecStart=/root/server-status/bin/python3 /root/server-status/server-status.py -c /root/server-status/server-status.yaml
Environment="PATH=/root/server-status/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
Restart=always
RestartSec=15s
KillMode=process
TimeoutStopSec=5s

[Install]
WantedBy=multi-user.target
```

Activate:
```bash
systemctl daemon-reload
systemctl enable --now server-status.service
journalctl -u server-status.service -f
```

---

## MQTT / Home Assistant

Each sensor is auto-discovered under the MQTT discovery prefix (default `homeassistant/`).  
Example topics:

```
SERVER/cpu_usage
SERVER/gpu/temp
SERVER/health_sda
SERVER/raid/md0
```

Availability topic: `<base>/availability`.

---

## Notes

- HDSentinel runs at most once every 30 minutes to avoid waking drives.  
- Percentages are published as integers.  
- GPU metrics require driver and `nvidia-smi` available in the systemd environment.  
- Tested on Debian 12 with Python 3.11 and NVIDIA 525+.  
