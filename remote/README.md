# Remote access

Turns the desktop into a headless AI server for other devices (e.g. a laptop running opencode),
reachable over Tailscale only. Run the [local setup](../README.md#setup) first.

```bash
sudo remote/install.sh
sudo tailscale up
```

`remote/install.sh` asks for each setting with a suggestion detected from the machine; set it in the
environment to skip the question:

| Variable | Default | |
|----------|---------|---|
| `RUN_USER` | `qwen` | service user, created if missing and hidden from the login screen |
| `SERVE_ARGS` | `--vision` (`--1bit` after a 1-bit install) | arguments for `serve.sh` |
| `OPENCODE_BIN` | the invoking user's opencode | copied to the service user |
| `OPENCODE_PORT` | `4096` | headless opencode server |
| `DOORBELL_PORT` | `45678` | UDP port that starts the stack |
| `WOL_IFACE` | interface of the default route | Wake-on-LAN |
| `TS_IFACE` | `tailscale0` | the only interface where the API ports are opened |

What it installs:

- `qwen-llama.service`: `serve.sh` listening on `0.0.0.0`, port from `config.env`.
- `qwen-opencode.service`: `opencode serve`, password in `/etc/qwen/server.env`.
- `qwen-doorbell.socket`: any UDP packet on the doorbell port starts `qwen.target` (both services).
- `wol.service`: enables Wake-on-LAN at every boot.
- ufw rules: the doorbell port, and the opencode and API ports on the Tailscale interface only.

From the laptop, or any always-on machine on the same LAN (e.g. a Raspberry Pi):

```bash
PC_MAC=<desktop MAC> PC_IP=<desktop LAN ip> PC_TS_IP=<desktop tailnet ip> remote/wake.sh
opencode --hostname <desktop tailnet ip> --port 4096
```

`wake.sh` sends a Wake-on-LAN magic packet (the sender must be on the desktop LAN), rings the
doorbell and waits until opencode answers on the tailnet IP. `remote/install.sh` prints the
command with the MAC and LAN IP filled in. It needs `python3` on the sender.

The API is also reachable directly at `http://<desktop tailnet ip>:8080/v1`.
