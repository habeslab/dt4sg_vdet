# IEC-104 Digital Twin Laboratory with Integrated GAN (E.Q.U.A.R.L.S.)

> **E.Q.U.A.R.L.S.** — *E quindi uscimmo a riveder le stelle*
> A comprehensive digital twin of an OT/ICS substation enhanced with a **Generative Adversarial Network (GAN)** for anomaly detection, renewable-grid interaction, and cyber-security research.

---

## 1. Purpose & Scope

This repository provides a reproducible, container-based digital twin of a medium-voltage electrical substation and its connected assets, extended with an **AI-driven GAN architecture**.

**Core objectives**

* **OT skill-building** — generate authentic IEC-104 traffic without risking live equipment.
* **Security assessment** — validate IDS signatures, detection use-cases and red-team playbooks in a safe sandbox.
* **GAN-based intrusion detection** — evaluate adversarial learning using NET-AGENT and NET-DISC.
* **Renewable integration** — model the effect of solar, wind and EV fast-charging on grid stability.
* **Rapid prototyping** — iterate on HMI/SCADA logic before field deployment.

The entire lab runs on **Docker Compose** (or can be imported into **GNS3**) and is made of the following nodes:

| Container/Role   | Hostname         | IPv4      | Function                                         |
| ---------------- | ---------------- | --------- | ------------------------------------------------ |
| **Master / HMI** | `master`         | 10.0.0.11 | Polls RTUs, issues set-points & commands         |
| **RTU Power**    | `rtu_power`      | 10.0.0.12 | Conventional thermal power plant                 |
| **RTU Factory**  | `rtu_factory`    | 10.0.0.13 | Heavy-industrial load                            |
| **RTU Suburb**   | `rtu_suburb`     | 10.0.0.14 | Residential feeders + battery storage            |
| **RTU Solar**    | `rtu_solar`      | 10.0.0.15 | Utility-scale photovoltaic farm                  |
| **RTU Wind**     | `rtu_wind`       | 10.0.0.16 | On-shore wind farm                               |
| **RTU Industry** | `rtu_industry`   | 10.0.0.17 | Industrial prosumer (generates & consumes)       |
| **RTU EV**       | `rtu_ev`         | 10.0.0.18 | EV fast-charging station                         |
| **Suricata IDS** | `ids` (host net) | —         | OT-aware intrusion detection system              |
| **Attacker**     | `attacker`       | 10.0.0.99 | Kali-like red-team box for adversarial scenarios |
| **NET-AGENT**    | `net-agent`      | 10.0.0.20 | GAN generator & sniffer                          |
| **NET-DISC**     | `disc-api`       | 10.0.0.21 | GAN discriminator REST API                       |

---

## 1.1 Protocol Stack

* **IEC 60870-5-104** over TCP (default **2404/tcp**).
* Optional **TLS** termination (disabled by default; enable via `ENABLE_TLS=true`).
* **JSON** log streams following the *EVE* schema (Suricata).

Each RTU periodically emits:

* **Spontaneous** single-point / analog updates (*M\_SP\_NA\_1*, *M\_ME\_NA\_1*).
* **Periodic** counter increments (*M\_CNT\_RC\_1*).

The Master executes a **General Interrogation** (*C\_IC\_NA\_1*) every 30 s (configurable) and processes set-point commands received via an internal CLI or REST endpoint.

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

## 2. GAN Integration

The project introduces a **Generative Adversarial Network (GAN)**:

* **NET-AGENT (Generator)** — captures IEC-104 flows and generates synthetic feature vectors.
* **NET-DISC (Discriminator)** — FastAPI service classifying flows into *benign*, *malicious*, *synthetic*.

**Loss Functions:**

* Generator:

$$
L_{GEN} = -\mathbb{E}_{z \sim p_z} [\log D(G(z))]
$$

* Discriminator:


$$ L_{DISC} = -\mathbb{E}_{x \sim p_{data}}[\log p_{label}(x)] $$ - $$ \mathbb{E}_{z \sim p_z}[\log (1 - p_{synthetic}(G(z)))] $$


**Operational Policy:**

* If $p_2 \geq \tau_{fake}$ ⇒ label = *synthetic*
* Else if $\frac{p_1}{p_0+p_1} \geq \tau_{mal}$ ⇒ label = *malicious*, otherwise *benign*.

### Monitoring NET-AGENT Logs

To inspect how the agent discriminates flows and forwards them to NET-DISC:

```bash
docker exec -it "net-agent" bash -lc 'ls -lh /data; tail -F /data/flows.jsonl 2>/dev/null | grep -a --line-buffered "flow_id"'
```

This command lists the JSONL logs and continuously streams classification results (`origin`, `label`, probabilities).

### Enabling the Generator

By default NET-AGENT operates in sniffer-only mode. To activate the GAN-based generator, set the environment variable:

```yaml
environment:
  - GEN_ENABLED=1
```

This enables the generation of synthetic feature vectors, which are dispatched alongside real traffic to NET-DISC.

---

## 3. Installation & Deployment

### Prerequisites

* **GNS3 ≥ 2.2** with Docker integration enabled
* **Docker Engine ≥ 24** plus the `docker compose` plugin
* \~**8 GB** of free RAM and **6 GB** of disk space

### Build & Run

```bash
cd iec104_lab
docker-compose build --no-cache
docker-compose up -d

cd gan-ics
docker-compose up -d
```

---

## 4. Importing the Twin into GNS3

The laboratory integrates with GNS3 in two flavours:

* **Docker Compose Import** — GNS3 parses docker-compose.yml and spawns the corresponding nodes.
* **Portable Project** — import a single .gns3project archive that embeds all images.

Consult the guides below for step-by-step instructions:

* [docs/import\_compose.md](iec104_lab/docs/import_compose.md)
* [docs/import\_portable.md](iec104_lab/docs/import_portable.md)

---

## 5. Launching Attack Scenarios

The **attacker** container provides `/attack.sh`, which guides you through available IEC-104 penetration tests:

```bash
docker compose exec attacker /attack.sh
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

## 6. Repository Structure

```
iec104_lab/
 ├── docker/           # RTU, Master, IDS, Attacker, Suricata
 ├── docs/             # Topology, guides
 └── project_portable/ # GNS3 portable project

gan-ics/
 ├── disc-api/         # Discriminator FastAPI
 ├── net-agent/        # Sniffer & Generator
 └── src/              # GAN training, preprocessing

utils/                 # Support Script

```

---

## 7. References

* Goodfellow et al. (2014) — *Generative Adversarial Nets*.
* IEC 60870-5-104 Standard.
* Mirsky et al. (2018) — *ICS Security & Anomaly Detection*.
* Digital Twin applications in Smart Grid Security.


