# Importing the Lab into GNS3 via **Portable Project**

This guide explains how to import the IEC‑104 Digital Twin laboratory as a **portable GNS3 project**, including all base images and topology, for rapid deployment. This method is ideal for demos, training environments, or when distributing the lab to colleagues.

---

## 1 — Portable project overview

A portable project is a single archive (`.portable`) containing:

* Node templates and configuration
* Topology wiring
* Optionally: Docker images (embedded)

Example file: `iec104_lab/project_portable/Digital_twin_iec104.portable`

---

## 2 — Import into GNS3

1. Launch **GNS3**.
2. Go to **File ▸ Import portable project…**.
3. Select the portable archive (`.portable`) provided.
4. Choose a destination folder for the project.
5. Finish the wizard → the full topology and nodes appear in your workspace.

> ✅ Within seconds, the entire lab (RTUs, Master, IDS, Attacker, plus GAN services **net-agent** and **disc-api**) is imported.

---

## 3 — Post‑import checks

* Verify node names and links match the intended topology (`iec104_lab/docs/topology.png`).
* Ensure **IDS** has 2 adapters: `eth0 → rtu_switch`, `eth1 → ot_switch`.
* If **net-agent** and **disc-api** are included, they behave like any other node. Use the same procedure as for RTUs or Master.

---

## 4 — Operating nodes

Open shells and execute commands directly:

```bash
# Open a shell in any node
docker exec -it <node> /bin/bash

# Launch the attack suite
docker exec -it attacker ./attack.sh

# Monitor Suricata logs
docker exec -it ids bash -lc 'tail -f /var/log/suricata/eve.json'
```

**Monitor NET‑AGENT classifications**:

```bash
docker exec -it net-agent bash -lc 'ls -lh /data; tail -F /data/flows.jsonl 2>/dev/null | grep -a --line-buffered "flow_id"'
```

**Enable the NET‑AGENT generator**:

```yaml
environment:
  - GEN_ENABLED_=1
```

Set this variable in the container template (or compose definition) and restart the node.

---

## 5 — Troubleshooting

* **Missing images**: If images were not embedded in the portable file, GNS3 prompts you to map them. Build them beforehand via Compose.
* **Adapters mismatch**: Edit the node template to adjust interface count.
* **Logs not visible**: Use `docker logs -f <node>` for debugging. Check NET‑AGENT `/data/flows.jsonl` for real‑time flow analysis.

---
