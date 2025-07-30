# IEC‑104 Digital‑Twin Laboratory (E.Q.U.A.R.L.S.)

> **E.Q.U.A.R.L.S.** — *E quindi uscimmo a riveder le stelle*  
> A comprehensive digital twin of an OT/ICS substation for research on IEC 60870‑5‑104, renewable‑grid interaction and cyber‑security.

## 1. Purpose & Scope

This repository provides a reproducible, container‑based twin of a medium‑voltage electrical substation and its connected assets.

**Core objectives**

* **OT skill‑building** — generate authentic IEC‑104 traffic without risking live equipment.
* **Security assessment** — validate IDS signatures, detection use‑cases and red‑team playbooks in a safe sandbox.
* **Renewable integration** — model the effect of solar, wind and EV fast‑charging on grid stability.
* **Rapid prototyping** — iterate on HMI/SCADA logic before field deployment.

The entire lab runs on **Docker Compose** (or can be imported into **GNS3**) and is made of the following nodes:

| Container/Role | Hostname | IPv4 | Function |
|----------------|----------|------|----------|
| **Master / HMI** | `master` | 10.0.0.11 | Polls RTUs, issues set‑points & commands |
| **RTU Power** | `rtu_power` | 10.0.0.12 | Conventional thermal power plant |
| **RTU Factory** | `rtu_factory` | 10.0.0.13 | Heavy‑industrial load |
| **RTU Suburb** | `rtu_suburb` | 10.0.0.14 | Residential feeders + battery storage |
| **RTU Solar** | `rtu_solar` | 10.0.0.15 | Utility‑scale photovoltaic farm |
| **RTU Wind** | `rtu_wind` | 10.0.0.16 | On‑shore wind farm |
| **RTU Industry** | `rtu_industry` | 10.0.0.17 | Industrial prosumer (generates & consumes) |
| **RTU EV** | `rtu_ev` | 10.0.0.18 | EV fast‑charging station |
| **Suricata IDS** | `ids` (host net) | — | OT‑aware intrusion detection system |
| **Attacker** | `attacker` | runtime | Kali‑like red‑team box for adversarial scenarios |

> **Topology** — see `iec104_lab/docs/topology.png` for an annotated diagram of VLANs, virtual switches and mirror ports.

### 1.1 Protocol Stack

* **IEC 60870‑5‑104** over TCP (default **2404/tcp**).
* Optional **TLS** termination (disabled by default; enable via `ENABLE_TLS=true`).
* **JSON** log streams following the *EVE* schema (Suricata).

Each RTU periodically emits:

* **Spontaneous** single‑point / analog updates (*M_SP_NA_1*, *M_ME_NA_1*).
* **Periodic** counter increments (*M_CNT_RC_1*).

The Master executes a **General Interrogation** (*C_IC_NA_1*) every 30 s (configurable) and processes set‑point commands received via an internal CLI or REST endpoint.

The Attacker container can craft malformed ASDUs or flood the network using `src/attack.sh`.

### 1.2 Addressing Scheme

| Attribute | Range | Defined in |
|-----------|-------|------------|
| **Common Address (CA)** | 1–7, unique per RTU | `CA` env var in `docker-compose.yml` |
| **Information Object Address (IOA)** | see table | JSON datasets |

| IOA Range | IEC‑104 Type | ASDU Code |
|-----------|--------------|-----------|
| 1–199   | Single‑Point Status | `M_SP_NA_1` |
| 200–399 | Double‑Point Status | `M_DP_NA_1` |
| 400–699 | Counter Values | `M_CNT_RC_1` |
| 700–899 | Analog Scaled | `M_ME_NA_1` |
| ≥900    | Reserved / Custom | — |

### 1.3 Communication Flow

```text
┌────────┐   General Call / Commands   ┌────────┐
│ Master │────────────────────────────▶│  RTU   │
│        │◀────────────────────────────│        │
└────────┘  Spontaneous Measurements   └────────┘

   Attacker ──► any node          IDS ◀── mirrored traffic / tap
```
## 4. Importing the Twin into GNS3

The laboratory integrates with GNS3 in two flavours:

Docker Compose Import — GNS3 parses docker-compose.yml and spawns the corresponding nodes.

Portable Project — import a single .gns3project archive that embeds all images.

Consult the guides below for step‑by‑step instructions:

* [docs/import_compose.md](iec104_lab/docs/import_compose.md)  

* [docs/import_portable.md](iec104_lab/docs/import_portable.md)