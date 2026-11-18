#!/usr/bin/env bash
set -euo pipefail

# Wakes the desktop (Wake-on-LAN magic packet), rings the doorbell that starts the stack and waits until opencode answers.
#   PC_MAC=<desktop MAC>  PC_IP=<desktop LAN ip>  PC_TS_IP=<desktop tailnet ip>
#   DOORBELL_PORT=45678  OPENCODE_PORT=4096  TRIES=100  INTERVAL=3

PC_MAC="${PC_MAC:?set PC_MAC to the MAC address of the desktop network interface}"
PC_IP="${PC_IP:?set PC_IP to the desktop LAN IP (give it a DHCP reservation)}"
PC_TS_IP="${PC_TS_IP:?set PC_TS_IP to the desktop tailnet IP (tailscale ip -4 on the desktop)}"
DOORBELL_PORT="${DOORBELL_PORT:-45678}"
OPENCODE_PORT="${OPENCODE_PORT:-4096}"
TRIES="${TRIES:-100}"
INTERVAL="${INTERVAL:-3}"

send_magic_packet() {
    python3 - "${PC_MAC}" <<'PY'
import socket, sys
mac = bytes.fromhex(sys.argv[1].replace(":", "").replace("-", ""))
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.sendto(b"\xff" * 6 + mac * 16, ("255.255.255.255", 9))
PY
}

for ((i = 0; i < TRIES; i++)); do
    if timeout 1 bash -c "exec 3<>/dev/tcp/${PC_TS_IP}/${OPENCODE_PORT}" 2>/dev/null; then
        echo "qwen stack is up"
        exit 0
    fi
    send_magic_packet
    printf 'wake' > "/dev/udp/${PC_IP}/${DOORBELL_PORT}" 2>/dev/null || true
    sleep "${INTERVAL}"
done

echo "the desktop did not come up after $((TRIES * INTERVAL)) s"
exit 1
