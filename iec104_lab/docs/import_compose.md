# Importing the Lab into GNS3 via **Docker Compose**

This guide explains how to build and register **all** containers — including the GAN services **net-agent** and **disc-api** — and then import them into **GNS3** using the **same procedure** adopted for the other nodes (RTUs, Master, IDS, Attacker).

---

## Prerequisites

* **GNS3 ≥ 2.2** with Docker integration enabled
* **Docker Engine ≥ 24** and the `docker compose` plugin
* \~**8 GB RAM**, **6 GB** disk space (minimum)
* (Optional) **GNS3 VM** configured with access to your local Docker daemon

---

## 1 — Build the images (Compose)

Build IEC‑104 lab nodes **and** GAN nodes.

```bash
# Build IEC‑104 Digital Twin images
cd iec104_lab
docker-compose build --no-cache

# Build GAN services (NET‑AGENT, DISC‑API)
cd ../gan-ics
docker-compose build --no-cache
```

> ✅ After this step, all images (rtu\_\*, master, ids, attacker, **net-agent**, **disc-api**) are available locally and can be registered in GNS3.

---

## 2 — Register images in GNS3 (same procedure for **all** nodes)

1. Open **GNS3** → **Edit ▸ Preferences ▸ Docker** → **New**.
2. Choose **Existing image** and select the tag you built.
3. Select **Run this Docker container on my local computer**.
4. **Adapters**: set **1** for every container **except `ids` (2 adapters)**.

   * `ids:eth0` → **rtu\_switch**
   * `ids:eth1` → **ot\_switch**
5. Leave **Start command** empty. Keep **Console** = **Telnet**.
6. Finish the wizard.

> ℹ️ **NET‑AGENT** and **DISC‑API** follow the **exact same registration flow** as the RTUs/Master/Attacker: select the existing image and complete the wizard. No special handling is required.

---

## 3 — Build the topology

1. From the device list, drag nodes to the workspace (RTUs, Master, IDS, Attacker, **net-agent**, **disc-api**).
2. Wire links according to `iec104_lab/docs/topology.png`.
3. Ensure `ids` has **two** links as noted above.
4. Place **net-agent** and **disc-api** on the monitoring/management network segment used by your lab.

> Tip: keep a consistent naming convention (e.g. `rtu_power`, `rtu_factory`, `net-agent`, `disc-api`).

---

## 4 — Start-up & sanity tests

After wiring is complete, save the project and press **▶ Start**.

```bash
# Open a shell inside a node
docker exec -it "node name" /bin/bash

# Launch the attack suite
docker exec -it attacker ./attack.sh

# Stream Suricata logs
docker exec -it ids bash -lc 'tail -f /var/log/suricata/eve.json'
```

**Monitor NET‑AGENT classifications** (real‑time, JSONL):

```bash
docker exec -it "net-agent" bash -lc 'ls -lh /data; tail -F /data/flows.jsonl 2>/dev/null | grep -a --line-buffered "flow_id"'
```

**Enable the generator** in NET‑AGENT (feature‑mode):

```yaml
# in the node env (Compose template or GNS3 Docker template)
environment:
  - GEN_ENABLED_=1
```

---

## 5 — Launch attack scenarios

The **attacker** node provides `/attack.sh` with prebuilt IEC‑104 tests:

```bash
docker compose exec attacker /attack.sh
# or (depending on node name):
docker exec -it attacker_node /attack.sh
```

**Menu options**

| Option               | Scenario          | Details                                                 |
| -------------------- | ----------------- | ------------------------------------------------------- |
| 1 — IDS test‑suite   | Synthetic attacks | Lateral movement, external intrusion, SYN‑flood         |
| 2 — Automatic replay | Dataset playback  | Replays every `*.pcap` in `/data` preserving timestamps |

**Runtime parameters**

| Variable       | Default | Purpose                                |
| -------------- | ------- | -------------------------------------- |
| `DELAY_FACTOR` | `1.0`   | Timing scale (0.5=2× speed, 2=½ speed) |
| `IFACE`        | `eth0`  | Scapy transmit interface               |

---

## 6 — Troubleshooting

* **Images not listed in GNS3** → restart GNS3 (and GNS3 VM, if used) after building.
* **Wrong number of adapters** → edit node template and set adapters as described.
* **No logs from NET‑AGENT** → verify container is running and `/data/flows.jsonl` exists; check env `GEN_` if you expect synthetic flows.
* **IDS not capturing** → confirm mirror link wiring and that Suricata is reading the correct interface.
