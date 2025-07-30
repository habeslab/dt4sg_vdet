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



