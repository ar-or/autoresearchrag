# Evaluator Container
# Debian-based Python with git, uv, codex-cli, and pre-downloaded data
#
# Usage:  docker compose up eval

FROM python:3.12-slim-bookworm

# Install system deps for codex, model runtimes, and native Python builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libgomp1 \
    nodejs \
    npm \
    openssh-client \
    pkg-config ripgrep jq fd-find git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install codex-cli globally
RUN npm install -g @openai/codex

# Set up SSH for GitHub deploy key
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null

# SSH config to use the deploy key for github.com
RUN printf 'Host github.com\n  IdentityFile /root/.ssh/github_deploy_key\n  StrictHostKeyChecking accept-new\n' > /root/.ssh/config \
    && chmod 600 /root/.ssh/config

# Set up codex config directory
RUN mkdir -p /root/.codex

# Entrypoint script: copies mounted secrets to writable locations with correct perms
RUN printf '#!/bin/bash\n\
if [ -f /mnt/secrets/github_deploy_key ]; then\n\
  cp /mnt/secrets/github_deploy_key /root/.ssh/github_deploy_key\n\
  chmod 600 /root/.ssh/github_deploy_key\n\
fi\n\
if [ -f /mnt/secrets/codex_auth.json ]; then\n\
  cp /mnt/secrets/codex_auth.json /root/.codex/auth.json\n\
  chmod 600 /root/.codex/auth.json\n\
fi\n\
exec "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

WORKDIR /app

# Copy project files needed for evaluators
COPY pyproject.toml uv.lock ./
COPY scripts/ scripts/
COPY evaluators/ evaluators/

# Install Python dependencies needed by the evaluators and retrieval-model experiments.
RUN uv venv && uv pip install \
    aiohttp \
    openai \
    python-dotenv \
    requests \
    sentencepiece \
    torch \
    transformers

# Pre-download evaluator datasets
RUN for script in evaluators/*/download_data.sh; do \
      [ -f "$script" ] && bash "$script"; \
    done

# Point ORAGENT_URL to host machine (--network host makes localhost work,
# but for non-host-network setups this is a useful default)
ENV ORAGENT_URL="http://host.docker.internal:32522"
ENV ELASTIC_URL="http://host.docker.internal:9200"

# Git config (for codex usage inside container)
RUN git config --global user.name "AI Agent Smith" \
    && git config --global user.email "alexander.raskin+ai1@openresearch.com"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
