# Importing the Lab into GNS3 via Docker Compose

## Prerequisites

* **GNS3 ≥ 2.2** with Docker integration enabled  
* **Docker Engine ≥ 24** plus the `docker compose` plugin  
* ~**8 GB** of free RAM and **6 GB** of disk space

## 1 – Build the images

    cd iec104_lab
    docker-compose build --no-cache

## 2 – Register the Docker images with GNS3

1. Launch **GNS3** and open **Edit ▸ Preferences ▸ Docker**.  
2. Click **New** for each image you have just built.  
3. Select **Run this Docker container on my local computer**  
   (the lab is designed for local execution, though you can use the **GNS3 VM** if preferred).  
4. Choose **Existing image** and pick the relevant tag from the list.  
5. Accept or edit the suggested name, then click **Next**.  
6. **Adapters**: set **1** for every container except **ids**, which requires **2 interfaces**  
   (*eth0* → **rtu_switch**, *eth1* → **ot_switch**).  
7. Leave **Start command** blank and click **Next**.  
8. Keep **Console type** as **Telnet**, then **Next** and **Finish** without adding environment variables.  

Repeat these steps for each container.

> **Tip:** When you have finished, restart GNS3 (and the GNS3 VM, if used) so the new nodes appear in the device list.

## 3 – Build the topology

1. In the left-hand panel, click **End devices** and drag the nodes onto the workspace.  
2. Connect the nodes following the diagram in `iec104_lab/docs/topology.png`.  
3. **IDS**: connect **eth0** to **rtu_switch** and **eth1** to **ot_switch**.

## 4 – Start-up and testing

After wiring is complete, save the project and press **▶ Start**.  
Within a few seconds all containers should be online:

    # Open a shell in an RTU
    docker exec -it  "node name" /bin/bash

     # Launch the attack script
    docker exec -it attacker ./attack.sh
    
    # Stream Suricata logs live
    tail -f /var/log/suricata/eve.json

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