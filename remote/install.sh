#!/usr/bin/env bash
set -euo pipefail

# Installs the headless stack: a service user running llama-server and opencode, a UDP wake
# doorbell, Wake-on-LAN and firewall rules that expose the API ports on Tailscale only.
# Asks for each setting with a detected suggestion; set it in the environment to skip the question.
#
#   RUN_USER=qwen  SERVE_ARGS=--vision  OPENCODE_BIN=<path>  OPENCODE_PORT=4096
#   DOORBELL_PORT=45678  WOL_IFACE=<default route interface>  TS_IFACE=tailscale0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "${SCRIPT_DIR}")"
UNITS_DIR="${SCRIPT_DIR}/units"

if [[ ${EUID} -ne 0 ]]; then
    echo "run with sudo" >&2
    exit 1
fi

ask() {
    local var="$1" prompt="$2" suggestion="$3" reply=""
    [[ -z "${!var:-}" ]] || return 0
    if [[ -t 0 ]]; then
        read -rp "${prompt} [${suggestion}]: " reply
    fi
    printf -v "${var}" '%s' "${reply:-${suggestion}}"
}

config_get() {
    local value=""
    [[ ! -f "${PROJ}/config.env" ]] || value="$(sed -n "s/^$1=//p" "${PROJ}/config.env" | tail -1)"
    echo "${value:-$2}"
}

INVOKING_HOME="$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)"
SUGGESTED_OPENCODE="${INVOKING_HOME}/.opencode/bin/opencode"
[[ -x "${SUGGESTED_OPENCODE}" ]] || SUGGESTED_OPENCODE="$(command -v opencode || true)"

ask RUN_USER "Service user" "qwen"
SUGGESTED_SERVE_ARGS="--vision"
[[ "$(config_get VARIANT full)" != "1bit" ]] || SUGGESTED_SERVE_ARGS="--1bit"
ask SERVE_ARGS "serve.sh arguments" "${SUGGESTED_SERVE_ARGS}"
ask OPENCODE_BIN "opencode binary" "${SUGGESTED_OPENCODE}"
ask OPENCODE_PORT "opencode server port" "4096"
ask DOORBELL_PORT "Wake doorbell UDP port" "45678"
ask WOL_IFACE "Wake-on-LAN network interface" "$(ip route show default 2>/dev/null | awk '{print $5; exit}')"
ask TS_IFACE "Tailscale interface" "tailscale0"
LLAMA_PORT="$(config_get PORT 8080)"
MODEL_DIR="$(config_get MODEL_DIR "${PROJ}/models")"
PROVIDER="$(config_get PROVIDER sqush)"
ETHTOOL="$(command -v ethtool || true)"

for cmd in jq setfacl openssl; do
    command -v "${cmd}" >/dev/null 2>&1 || { echo "!! ${cmd} not found" >&2; exit 1; }
done
[[ -x "${OPENCODE_BIN}" ]] || { echo "!! opencode binary not found: ${OPENCODE_BIN:-<empty>}" >&2; exit 1; }
[[ -n "${ETHTOOL}" ]] || { echo "!! ethtool not found (e.g. sudo apt install ethtool)" >&2; exit 1; }
[[ -n "${WOL_IFACE}" ]] || { echo "!! no network interface for Wake-on-LAN, set WOL_IFACE" >&2; exit 1; }

if ! id "${RUN_USER}" &>/dev/null; then
    echo "==> creating the headless user ${RUN_USER}"
    useradd -r -m -s /usr/sbin/nologin "${RUN_USER}"
fi
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"

echo "==> hiding ${RUN_USER} from the login screen"
mkdir -p /var/lib/AccountsService/users
printf '[User]\nSystemAccount=true\n' > "/var/lib/AccountsService/users/${RUN_USER}"

echo "==> opencode server password"
mkdir -p /etc/qwen
if [[ ! -f /etc/qwen/server.env ]]; then
    (umask 077; printf 'OPENCODE_SERVER_PASSWORD=%s\n' "$(openssl rand -hex 24)" > /etc/qwen/server.env)
fi
chmod 600 /etc/qwen/server.env
echo "    saved in /etc/qwen/server.env"

# The service user needs to traverse every parent directory to reach the project.
grant() {
    local mode="$1" path="$2" p
    p="$(dirname "${path}")"
    while [[ "${p}" != "/" ]]; do
        setfacl -m "u:${RUN_USER}:x" "${p}"
        p="$(dirname "${p}")"
    done
    setfacl -R -m "u:${RUN_USER}:${mode}" "${path}"
}
echo "==> granting ${RUN_USER} access to ${PROJ} and ${MODEL_DIR}"
grant rX "${PROJ}"
mkdir -p "${MODEL_DIR}"
grant rwX "${MODEL_DIR}"

echo "==> installing opencode for ${RUN_USER}"
install -d -o "${RUN_USER}" -g "${RUN_USER}" "${RUN_HOME}/.opencode/bin" "${RUN_HOME}/.config/opencode"
install -o "${RUN_USER}" -g "${RUN_USER}" -m 755 "${OPENCODE_BIN}" "${RUN_HOME}/.opencode/bin/opencode"
jq --arg prov "${PROVIDER}" --arg base "http://127.0.0.1:${LLAMA_PORT}/v1" \
    '.provider |= with_entries(.key = $prov | .value.options.baseURL = $base)' \
    "${PROJ}/opencode.json" > "${RUN_HOME}/.config/opencode/opencode.json"
chown "${RUN_USER}:${RUN_USER}" "${RUN_HOME}/.config/opencode/opencode.json"

echo "==> installing the systemd units"
for unit in "${UNITS_DIR}"/*; do
    sed -e "s|@PROJ@|${PROJ}|g" \
        -e "s|@RUN_USER@|${RUN_USER}|g" \
        -e "s|@RUN_HOME@|${RUN_HOME}|g" \
        -e "s|@SERVE_ARGS@|${SERVE_ARGS}|g" \
        -e "s|@OPENCODE_PORT@|${OPENCODE_PORT}|g" \
        -e "s|@DOORBELL_PORT@|${DOORBELL_PORT}|g" \
        -e "s|@WOL_IFACE@|${WOL_IFACE}|g" \
        -e "s|@ETHTOOL@|${ETHTOOL}|g" \
        "${unit}" > "/etc/systemd/system/$(basename "${unit}")"
    chmod 644 "/etc/systemd/system/$(basename "${unit}")"
done
systemctl daemon-reload
systemctl enable --now qwen-doorbell.socket
systemctl enable wol.service

if command -v ufw >/dev/null 2>&1; then
    echo "==> firewall"
    ufw allow "${DOORBELL_PORT}/udp" >/dev/null
    ufw allow in on "${TS_IFACE}" to any port "${OPENCODE_PORT}" proto tcp >/dev/null
    ufw allow in on "${TS_IFACE}" to any port "${LLAMA_PORT}" proto tcp >/dev/null
else
    echo "!! ufw not found: allow ${DOORBELL_PORT}/udp, and tcp ${OPENCODE_PORT} and ${LLAMA_PORT} on ${TS_IFACE} only"
fi

echo "==> Wake-on-LAN on ${WOL_IFACE} (now and at every boot)"
"${ETHTOOL}" -s "${WOL_IFACE}" wol g
"${ETHTOOL}" "${WOL_IFACE}" | grep -i 'wake-on' || true

if ! command -v tailscale >/dev/null 2>&1; then
    echo "==> installing tailscale"
    curl -fsSL https://tailscale.com/install.sh | sh
fi

echo
echo "Done. Next steps:"
echo "  sudo tailscale up && tailscale ip -4"
echo "  from the laptop: opencode --hostname <desktop tailnet ip> --port ${OPENCODE_PORT}"
echo "  to wake it from another machine:"
echo "    PC_MAC=$(cat "/sys/class/net/${WOL_IFACE}/address") PC_IP=$(ip -4 -o addr show "${WOL_IFACE}" | awk '{sub("/.*", "", $4); print $4; exit}') PC_TS_IP=<tailnet ip> \\"
echo "    DOORBELL_PORT=${DOORBELL_PORT} OPENCODE_PORT=${OPENCODE_PORT} remote/wake.sh"
