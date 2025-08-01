## 2 — Import into GNS3 via portable project

1. Launch **GNS3** and create a **New Project** (for example *iec104_lab*).  
2. Navigate to **File ▸ Import portable project…**  
3. Select the freshly downloaded `iec104_lab\project_portable\Digital_twin_iec104.portable` and click **Open**.  
4. Choose (or confirm) the destination folder for the project.  
5. Click the **Start** (►) button in the toolbar to power on all nodes.

> ✅ Within seconds, the containers will be online and the links will turn green.

## 3 — Working with the containers

| Task | Command |
|------|---------|
| Open a shell inside an RTU | `docker exec -it  "node name" /bin/bash` |
| Run the attack script      | ` docker exec -it attacker ./attack.sh`  |
| Open a shell in an RTU     | `docker exec -it  "ids" /bin/bash`       |
| Stream Suricata logs live  | `tail -f /var/log/suricata/eve.json`     |

## Launching Attack Scenarios

The **attacker** container provides the control script `/attack.sh`,  
which guides you through the available IEC-104 penetration tests.

    docker compose exec attacker /attack.sh
    # or, if the container name differs:
    docker exec -it attacker_node /attack.sh

### Menu options

| Option | Scenario            | Operational details |
|--------|---------------------|---------------------|
| **1 – IDS test-suite** | *Synthetic attacks* | Generates three demonstration flows (lateral movement, external intrusion, SYN-flood) to validate the installed IDS/IPS rules. |
| **2 – Automatic replay** | *Dataset playback* | Sequentially replays — preserving the original timing — **all** `*.pcap` files located in **/data**. The engine is dataset-agnostic: any **IEC-104 over TCP** capture placed in `/data` is detected and injected, with no constraints on file names or folder structure. |

### Runtime parameters

| Variable        | Default | Purpose |
|-----------------|---------|---------|
| `DELAY_FACTOR`  | `1.0`   | Timing scale for the replay (`0.5` = twice as fast, `2` = half-speed). |
| `IFACE`         | `eth0`  | Network interface used by Scapy to transmit the frames. |
