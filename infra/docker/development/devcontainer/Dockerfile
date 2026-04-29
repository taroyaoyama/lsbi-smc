FROM node:24

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

ARG USER_UID=1000
ARG USERNAME=node
RUN if [ "$USER_UID" != "1000" ]; then usermod -u $USER_UID $USERNAME; fi

# Install basic development tools and iptables/ipset
RUN apt-get update && apt-get install -y --no-install-recommends \
    less \
    git \
    procps \
    sudo \
    fzf \
    zsh \
    man-db \
    unzip \
    gnupg2 \
    gh \
    iptables \
    ipset \
    iproute2 \
    dnsutils \
    aggregate \
    jq \
    nano \
    vim \
    tcpdump \
    python3-minimal \
    rsyslog \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure default node user has access to /usr/local/share
RUN mkdir -p /usr/local/share/npm-global && \
    chown -R node:node /usr/local/share

# Persist bash history.
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory

# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true

ARG REPO_NAME

# Create workspace and config directories and set permissions
RUN mkdir -p /workspace/${REPO_NAME} /home/node/.claude && \
    chown -R node:node /workspace/${REPO_NAME} /home/node/.claude

WORKDIR /workspace/${REPO_NAME}

ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
    wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    sudo dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# Set up non-root user
USER node

# Install global packages
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Set the default shell to zsh rather than sh
ENV SHELL=/bin/zsh

# Set the default editor and visual
ENV EDITOR=nano
ENV VISUAL=nano

# Default powerline10k theme
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
    -p git \
    -p fzf \
    -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
    -a "source /usr/share/doc/fzf/examples/completion.zsh" \
    -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    -x

# Install Claude
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}


# Copy and set up firewall script
COPY ./infra/docker/development/devcontainer/init-firewall.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh && \
    printf 'node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh\nnode ALL=(root) NOPASSWD: /bin/dmesg\n' \
        > /etc/sudoers.d/node-firewall && \
    chmod 0440 /etc/sudoers.d/node-firewall
USER node

# Under this, customize the image as needed, e.g., install additional tools, set up configurations, etc.

# Install UV and Python 3.12
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
RUN uv python install 3.12

# Copy and set up sync script
COPY ./infra/docker/development/devcontainer/background-sync.sh /usr/local/bin/
COPY ./infra/docker/development/devcontainer/fix-venv-permissions.sh /usr/local/bin/
COPY ./infra/docker/development/devcontainer/sni-capture.py /usr/local/bin/
COPY ./infra/docker/development/devcontainer/fw-show-blocked.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/background-sync.sh /usr/local/bin/fix-venv-permissions.sh \
             /usr/local/bin/sni-capture.py /usr/local/bin/fw-show-blocked.sh && \
    usermod -aG adm node && \
    echo "node ALL=(root) NOPASSWD: /usr/local/bin/fix-venv-permissions.sh" > /etc/sudoers.d/node-background-sync && \
    chmod 0440 /etc/sudoers.d/node-background-sync
USER node

# Entrypoint: runs as root to initialize firewall before main process
COPY ./infra/docker/development/devcontainer/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
USER root
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["sleep", "infinity"]
