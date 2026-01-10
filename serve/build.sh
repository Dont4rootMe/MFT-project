#!/bin/bash
# Build script for TorchServe A2C Trading Model
# 
# This script:
# 1. Exports model from outputs/ checkpoint
# 2. Builds Docker image with full project (pipelines/, conf/, tensortrade/)
#
# Usage:
#   ./serve/build.sh                    # Build with default checkpoint
#   ./serve/build.sh /path/to/checkpoint # Build with custom checkpoint

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHECKPOINT="${1:-outputs/model/checkpoints/latest}"
MODEL_NAME="a2c-trading"

echo "🚀 Building TorchServe Docker image for A2C Trading Model"
echo "   Project root: $PROJECT_ROOT"
echo "   Checkpoint: $CHECKPOINT"
echo ""

cd "$PROJECT_ROOT"

# Step 1: Export model to .mar archive
echo "📦 Step 1: Exporting model from checkpoint..."
python serve/export_model.py \
    --checkpoint "$CHECKPOINT" \
    --output serve/model-store \
    --name "$MODEL_NAME"

# Step 2: Prepare context for Docker build
# Docker context needs access to pipelines/, conf/, tensortrade/
echo ""
echo "📁 Step 2: Preparing Docker build context..."

# Create a temporary build context
BUILD_CONTEXT="$SCRIPT_DIR/build-context"
rm -rf "$BUILD_CONTEXT"
mkdir -p "$BUILD_CONTEXT"

# Copy necessary files
cp -r pipelines/ "$BUILD_CONTEXT/pipelines/"
cp -r conf/ "$BUILD_CONTEXT/conf/"
cp -r tensortrade/ "$BUILD_CONTEXT/tensortrade/"
cp -r serve/model-store/ "$BUILD_CONTEXT/model-store/"
cp serve/config.properties "$BUILD_CONTEXT/config.properties"
cp serve/Dockerfile "$BUILD_CONTEXT/Dockerfile"

echo "   ✓ Copied pipelines/, conf/, tensortrade/, model-store/"

# Step 3: Build Docker image
echo ""
echo "🐳 Step 3: Building Docker image..."
docker build -t a2c-serve:v1 "$BUILD_CONTEXT"

# Cleanup
rm -rf "$BUILD_CONTEXT"

echo ""
echo "✅ Build complete!"
echo ""
echo "To run the service:"
echo "  docker run -d -p 8080:8080 -p 8081:8081 -p 8082:8082 --name a2c-serve a2c-serve:v1"
echo ""
echo "To test:"
echo "  curl -X POST http://localhost:8080/predictions/a2c-trading -T serve/sample_input.json"
echo ""
echo "To view logs:"
echo "  docker logs -f a2c-serve"
echo ""
echo "To stop:"
echo "  docker stop a2c-serve && docker rm a2c-serve"
