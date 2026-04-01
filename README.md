# Explainable Security Monitoring for Smart Grid Infrastructures via a Network Digital Twin
### A comprehensive digital twin of an OT/ICS substation enhanced with a ML-based IDS for anomaly detection.

---

This repository provides a reproducible, container-based architecture for the monitoring and simulation of a Smart Grid infrastructure and its connected assets, extended with a ML-based and XAI architecture.

The entire lab runs on **Docker Compose** (or can be imported into **GNS3**) and is made of the following nodes:

| Container/Role   | Hostname         | IPv4      | Function                                         |
| ---------------- | ---------------- | --------- | ------------------------------------------------ |
| __Master / HMI__ | `master`         | 10.0.0.11 | Polls RTUs, issues set-points & commands         |
| __RTU Power__    | `rtu_power`      | 10.0.0.12 | Conventional thermal power plant                 |
| __RTU Factory__  | `rtu_factory`    | 10.0.0.13 | Heavy-industrial load                            |
| __RTU Suburb__   | `rtu_suburb`     | 10.0.0.14 | Residential feeders + battery storage            |
| __RTU Solar__    | `rtu_solar`      | 10.0.0.15 | Utility-scale photovoltaic farm                  |
| __RTU Wind__     | `rtu_wind`       | 10.0.0.16 | On-shore wind farm                               |
| __RTU Industry__ | `rtu_industry`   | 10.0.0.17 | Industrial prosumer (generates & consumes)       |
| __RTU EV__       | `rtu_ev`         | 10.0.0.18 | EV fast-charging station                         |
| __Suricata IDS__ | `ids` (host net) | —         | OT-aware intrusion detection system              |
| __Attacker__     | `attacker`       | 10.0.0.99 | Kali-like red-team box for adversarial scenarios |
| __NET-AGENT__    | `net-agent`      | 10.0.0.20 | Serves as ML-based classifier
| __NET-DISC__     | `disc-api`       | 10.0.0.21 | REST API invoking the net agent                       |

---

## 1.1 Protocol Stack

* **IEC 60870-5-104** over TCP (default **2404/tcp**).
* Optional __TLS__ termination (disabled by default; enable via `ENABLE_TLS=true`).
* **JSON** log streams following the *EVE* schema (Suricata).

Each RTU periodically emits:

* __Spontaneous__ single-point / analog updates (_M\_SP\_NA\_1_, _M\_ME\_NA\_1_).
* __Periodic__ counter increments (_M\_CNT\_RC\_1_).

The Master executes a __General Interrogation__ (_C\_IC\_NA\_1_) every 30 s (configurable) and processes set-point commands received via an internal CLI or REST endpoint.

The Attacker container can craft malformed ASDUs or flood the network using `attack.sh` or replay PCAP datasets.

---

## 1.2 Addressing Scheme

| Attribute                            | Range               | Defined in                           |
| ------------------------------------ | ------------------- | ------------------------------------ |
| **Common Address (CA)**              | 1–7, unique per RTU | `CA` env var in `docker-compose.yml` |
| **Information Object Address (IOA)** | see table           | JSON datasets                        |

| IOA Range | IEC-104 Type        | ASDU Code    |
| --------- | ------------------- | ------------ |
| 1–199     | Single-Point Status | `M_SP_NA_1`  |
| 200–399   | Double-Point Status | `M_DP_NA_1`  |
| 400–699   | Counter Values      | `M_CNT_RC_1` |
| 700–899   | Analog Scaled       | `M_ME_NA_1`  |
| ≥900      | Reserved / Custom   | —            |

---

## 1.3 Communication Flow

```text
┌────────┐   General Call / Commands   ┌────────┐
│ Master │────────────────────────────▶│  RTU   │
│        │◀────────────────────────────│        │
└────────┘  Spontaneous Measurements   └────────┘

   Attacker ──► any node          IDS ◀── mirrored traffic / tap

   NET-AGENT (Sniffer/Generator) ──► Features ──► NET-DISC (API)
```

---

## 2. ML Integration

The project introduces a **ML-based IDS**:

* **Net-Agent** — captures IEC-104 flows and aggregates packets.
* **ML/XAI Layer** — FastAPI service classifying flows into *Normal*, *Integrity*, *Availability* (attacks).

### Monitoring NET-AGENT Logs

To inspect how the agent discriminates flows and forwards them to NET-DISC:

```bash
docker logs -it "net-agent"
```

---

## 3. Installation & Deployment

### Prerequisites

* **GNS3 ≥ 2.2** with Docker integration enabled
* **Docker Engine ≥ 24** plus the `docker compose` plugin
* \~**8 GB** of free RAM and **6 GB** of disk space

### Build & Run

```bash
cd iec104_lab
docker compose build

cd gan-ics
docker compose build

cd mqtt
docker compose build

cd physical_nodes
docker compose build
```

---

## 4. Importing the Twin into GNS3

To start up the simulation environment into GNS3, you need to import the image file (`gns3/dt4sg.gns3project`), and then to associate the docker images with the nodes. At this stage, if you don't have previously build the docker images, you might fail in having the association working.

---

## 5. Launching Attack Scenarios

Once inialized the simulation, the **attacker** container provides `/attack.sh`, which guides you through available IEC-104 penetration tests:

```bash
docker exec "attacker" bash 
./attack.sh
```

**Menu options:**

| Option                   | Scenario            | Operational details                                                             |
| ------------------------ | ------------------- | ------------------------------------------------------------------------------- |
| **1 – IDS test-suite**   | *Synthetic attacks* | Generates flows (lateral movement, intrusion, SYN-flood) to validate IDS.       |
| **2 – Automatic replay** | *Dataset playback*  | Sequentially replays all `*.pcap` files in `/data`, preserving original timing. |

**Runtime Parameters:**

| Variable       | Default | Purpose                          |
| -------------- | ------- | -------------------------------- |
| `DELAY_FACTOR` | `1.0`   | Timing scale (`0.5` = 2x speed). |
| `IFACE`        | `eth0`  | Interface for replay.            |

---


