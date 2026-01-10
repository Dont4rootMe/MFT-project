# TorchServe Dockerfile for A2C Trading Model
# Uses the original project code from pipelines/ and conf/

FROM pytorch/torchserve:latest-cpu

# Set environment variables
ENV MODEL_NAME=a2c-trading
ENV MODEL_STORE=/home/model-server/model-store
ENV PYTHONPATH=/app

# Switch to root for installation
USER root

# Install additional dependencies
RUN pip install --no-cache-dir numpy pandas hydra-core omegaconf quantstats

# Create directories
RUN mkdir -p ${MODEL_STORE} /app

# Copy the entire project for imports (pipelines/, conf/, tensortrade/)
COPY --chown=model-server:model-server pipelines/ /app/pipelines/
COPY --chown=model-server:model-server conf/ /app/conf/
COPY --chown=model-server:model-server tensortrade/ /app/tensortrade/

# Copy model archive to model-store
COPY --chown=model-server:model-server model-store/${MODEL_NAME}.mar ${MODEL_STORE}/

# Copy TorchServe configuration
COPY --chown=model-server:model-server config.properties /home/model-server/config.properties

# Switch back to model-server user
USER model-server

# Expose ports
# 8080: Inference API
# 8081: Management API  
# 8082: Metrics API
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# Start TorchServe
CMD ["torchserve", \
     "--start", \
     "--model-store", "/home/model-server/model-store", \
     "--models", "a2c-trading=a2c-trading.mar", \
     "--ts-config", "/home/model-server/config.properties", \
     "--foreground"]
