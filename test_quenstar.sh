#!/usr/bin/env bash
# test_quenstar.sh — End-to-end test suite for QuenStar inference
# Tests: server startup, single request, streaming, concurrent requests
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_LOG="/tmp/quenstar_test.log"
API_URL="http://127.0.0.1:9898/v1/chat/completions"

pass() { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${NC} $1 — $2"; FAIL=$((FAIL + 1)); }

cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -f "run.sh serve" 2>/dev/null || true
    pkill -f "quenstar" 2>/dev/null || true
    sleep 2
    rm -f "$SERVER_LOG"
}
trap cleanup EXIT

start_server() {
    echo -e "${YELLOW}[STARTING]${NC} QuenStar server..."
    rm -f "$SERVER_LOG"
    bash "$SCRIPT_DIR/run.sh" serve > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"

    # Wait for server to be ready (up to 2 minutes)
    for i in $(seq 1 120); do
        if grep -q "Uvicorn running" "$SERVER_LOG" 2>/dev/null; then
            echo -e "  ${GREEN}Server ready${NC} after ${i}s"
            return 0
        fi
        if grep -qi "traceback\|error.*fatal\|OOM" "$SERVER_LOG" 2>/dev/null; then
            echo -e "  ${RED}Server crashed during startup${NC}"
            tail -20 "$SERVER_LOG"
            return 1
        fi
        sleep 1
    done
    echo -e "  ${RED}Timeout waiting for server${NC}"
    return 1
}

test_single_request() {
    echo -e "\n${YELLOW}[TEST]${NC} Single chat request"
    local response
    response=$(curl -s --max-time 120 -X POST "$API_URL" \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":5}' 2>&1) || true
    if echo "$response" | grep -q '"choices"'; then
        pass "Got valid JSON response"
    else
        fail "No choices in response" "$(echo "$response" | head -c 200)"
    fi
}

test_streaming_request() {
    echo -e "\n${YELLOW}[TEST]${NC} Streaming chat request"
    local response
    response=$(curl -s --max-time 120 -N -X POST "$API_URL" \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Say test"}],"max_tokens":10,"stream":true}' 2>&1) || true
    if echo "$response" | grep -q '"choices"'; then
        pass "Got streaming response with choices"
    else
        fail "No streaming data" "$(echo "$response" | head -c 200)"
    fi
}

test_concurrent_requests() {
    echo -e "\n${YELLOW}[TEST]${NC} 3 concurrent requests (stress test)"
    local pids=()
    local results=()
    for i in $(seq 1 3); do
        curl -s --max-time 180 -X POST "$API_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":10}" \
            > "/tmp/quenstar_result_$i.json" 2>/dev/null &
        pids+=($!)
    done

    local all_ok=true
    for i in $(seq 1 3); do
        wait "${pids[$((i-1))]}" 2>/dev/null || true
        if grep -q '"choices"' "/tmp/quenstar_result_$i.json" 2>/dev/null; then
            : # ok
        else
            all_ok=false
            echo "  Request $i failed: $(head -c 100 /tmp/quenstar_result_$i.json 2>/dev/null)"
        fi
        rm -f "/tmp/quenstar_result_$i.json"
    done

    if $all_ok; then
        pass "All 3 concurrent requests succeeded"
    else
        fail "Some concurrent requests failed" "check Thread errors in server log"
    fi
}

test_server_errors() {
    echo -e "\n${YELLOW}[TEST]${NC} Server error log check"
    # Check for crashes in Thread-N
    if grep -q "Exception in thread" "$SERVER_LOG" 2>/dev/null; then
        fail "Server thread crashed" "$(grep 'Exception in thread' "$SERVER_LOG" | tail -1)"
    else
        pass "No thread crashes in server log"
    fi

    # Check for CUDA OOM
    if grep -qi "out.*memory\|OOM" "$SERVER_LOG" 2>/dev/null; then
        fail "CUDA OOM detected" "$(grep -i 'out.*memory' "$SERVER_LOG" | tail -1)"
    else
        pass "No CUDA OOM"
    fi

    # Check for triton autotune errors
    if grep -q "TypeError.*NoneType.*mapping\|RuntimeError.*No valid config" "$SERVER_LOG" 2>/dev/null; then
        fail "Triton autotune crash" "$(grep 'TypeError.*NoneType\|RuntimeError.*config' "$SERVER_LOG" | tail -1)"
    else
        pass "No triton autotune crashes"
    fi
}

# ── Main ────────────────────────────────────────────────────
echo "============================================"
echo " QuenStar End-to-End Test Suite"
echo "============================================"

# 1. Start server
if ! start_server; then
    echo -e "\n${RED}Server failed to start — aborting${NC}"
    exit 1
fi

# 2. Run tests
test_single_request
test_streaming_request
test_concurrent_requests
test_server_errors

# 3. Summary
echo -e "\n============================================"
echo -e " Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "============================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
