#!/usr/bin/env bash
# test_quantstar.sh — End-to-end test suite for QuantStar inference
# Tests: server startup, single request, streaming, concurrent requests
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_LOG="/tmp/quantstar_test.log"
API_URL="http://127.0.0.1:9898/v1/chat/completions"

pass() { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${NC} $1 — $2"; FAIL=$((FAIL + 1)); }

cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -f "run.sh serve" 2>/dev/null || true
    pkill -f "quantstar" 2>/dev/null || true
    sleep 2
    rm -f "$SERVER_LOG"
}
trap cleanup EXIT

start_server() {
    echo -e "${YELLOW}[STARTING]${NC} QuantStar server..."
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
            > "/tmp/quantstar_result_$i.json" 2>/dev/null &
        pids+=($!)
    done

    local all_ok=true
    for i in $(seq 1 3); do
        wait "${pids[$((i-1))]}" 2>/dev/null || true
        if grep -q '"choices"' "/tmp/quantstar_result_$i.json" 2>/dev/null; then
            : # ok
        else
            all_ok=false
            echo "  Request $i failed: $(head -c 100 /tmp/quantstar_result_$i.json 2>/dev/null)"
        fi
        rm -f "/tmp/quantstar_result_$i.json"
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

test_reasoning_leak() {
    echo -e "\n${YELLOW}[TEST]${NC} Reasoning leak check (complex math problem)"
    local response
    response=$(curl -s --max-time 180 -N -X POST "$API_URL" \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"If a train leaves at 3pm traveling 60mph and another leaves at 4pm traveling 80mph, when do they meet if they are 300 miles apart? Solve step by step."}],"max_tokens":200,"stream":true,"temperature":0.7}' 2>&1) || true

    # Check if model generated any thinking at all
    local has_think_tags=false
    if echo "$response" | grep -q '<think>\|</think>' 2>/dev/null; then
        has_think_tags=true
    fi

    if $has_think_tags; then
        # Model generated thinking — verify it's correctly parsed as reasoning_content
        local has_reasoning=false
        if echo "$response" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: ') and line != 'data: [DONE]':
        d = json.loads(line[6:])
        c = d.get('choices', [{}])[0]
        delta = c.get('delta', {})
        if delta.get('reasoning_content') or delta.get('reasoning_text'):
            sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            has_reasoning=true
        fi

        # Check <think> or </think> NOT leaked as content
        local leaked=false
        if echo "$response" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: ') and line != 'data: [DONE]':
        d = json.loads(line[6:])
        c = d.get('choices', [{}])[0]
        delta = c.get('delta', {})
        content = delta.get('content', '')
        if '<think>' in content or '</think>' in content:
            sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            leaked=true
        fi

        if $has_reasoning; then
            pass "Reasoning emitted as reasoning_content/reasoning_text"
        else
            fail "Model generated <think> but not emitted as reasoning"
        fi

        if $leaked; then
            fail "<think> or </think> leaked as content"
        else
            pass "No <think> leak in content"
        fi
    else
        pass "Model did not generate thinking (skipped leak check)"
    fi
}

# ── Main ────────────────────────────────────────────────────
echo "============================================"
echo " QuantStar End-to-End Test Suite"
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
test_reasoning_leak
test_server_errors

# 3. Summary
echo -e "\n============================================"
echo -e " Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "============================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
