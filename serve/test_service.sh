#!/bin/bash
# Test script for TorchServe A2C Trading Model
#
# Usage:
#   ./serve/test_service.sh              # Test with default endpoint
#   ./serve/test_service.sh localhost    # Test specific host
#   ./serve/test_service.sh 192.168.1.1  # Test remote host

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-localhost}"
PORT="${2:-8080}"
MODEL_NAME="a2c-trading"
BASE_URL="http://${HOST}:${PORT}"

echo "🧪 Testing TorchServe A2C Trading Model"
echo "   Endpoint: ${BASE_URL}"
echo ""

# Test 1: Health check
echo "📍 Test 1: Health check (ping)..."
PING_RESULT=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/ping" 2>/dev/null || echo "000")
if [ "$PING_RESULT" = "200" ]; then
    echo "   ✅ Service is healthy"
else
    echo "   ❌ Service not responding (HTTP $PING_RESULT)"
    echo "   Is the container running? Try: docker ps"
    exit 1
fi

# Test 2: Model info (management API)
echo ""
echo "📍 Test 2: Model info..."
MODEL_INFO=$(curl -s "http://${HOST}:8081/models/${MODEL_NAME}" 2>/dev/null || echo "error")
if [[ "$MODEL_INFO" != "error" ]]; then
    echo "   ✅ Model registered:"
    echo "$MODEL_INFO" | python3 -m json.tool 2>/dev/null || echo "$MODEL_INFO"
else
    echo "   ⚠️ Could not get model info (management API may be on different port)"
fi

# Test 3: Inference with sample input
echo ""
echo "📍 Test 3: Inference test..."
PREDICTION=$(curl -s -X POST "${BASE_URL}/predictions/${MODEL_NAME}" \
    -H "Content-Type: application/json" \
    -d @"${SCRIPT_DIR}/sample_input.json" 2>/dev/null || echo "error")

if [[ "$PREDICTION" != "error" && "$PREDICTION" != "" ]]; then
    echo "   ✅ Prediction received:"
    echo "$PREDICTION" | python3 -m json.tool 2>/dev/null || echo "$PREDICTION"
else
    echo "   ❌ Inference failed"
    exit 1
fi

# Test 4: Metrics (if enabled)
echo ""
echo "📍 Test 4: Metrics endpoint..."
METRICS=$(curl -s "http://${HOST}:8082/metrics" 2>/dev/null | head -20 || echo "")
if [ -n "$METRICS" ]; then
    echo "   ✅ Metrics available (showing first 5 lines):"
    echo "$METRICS" | head -5
else
    echo "   ⚠️ Metrics endpoint not responding"
fi

echo ""
echo "✅ All tests completed!"

