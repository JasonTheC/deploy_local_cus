#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════
# CUS Backend — One-Line Deploy (Arch Linux / pacman)
# Usage:  ./deploy.sh              # auto-install Docker, then build + start
#         ./deploy.sh install-docker  # install Docker Engine + Compose only
#         ./deploy.sh install-gpu     # install nvidia-container-toolkit
#         ./deploy.sh stop            # stop all containers
#         ./deploy.sh logs            # follow logs
#         ./deploy.sh status          # show running services + URLs
# ═══════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_BASE_FILE="$SCRIPT_DIR/docker-compose.yml"
COMPOSE_GPU_FILE="$SCRIPT_DIR/docker-compose.gpu.yml"
COMPOSE_FILES=(-f "$COMPOSE_BASE_FILE")
GPU_AVAILABLE=0

compose() {
  docker compose "${COMPOSE_FILES[@]}" "$@"
}

# ── Install Docker on Arch Linux via pacman ────────────────
install_docker_arch() {
  echo "▶ Installing Docker Engine + Compose plugin (pacman)…"
  sudo pacman -Sy --noconfirm docker docker-compose-plugin
  sudo systemctl enable --now docker
  # Add current user to docker group so future runs don't need sudo
  sudo usermod -aG docker "$USER" 2>/dev/null || true
  echo "✓ Docker installed and started."
}

# ── Install NVIDIA Container Toolkit (optional GPU support) ─
install_nvidia_toolkit_arch() {
  echo "▶ Installing nvidia-container-toolkit for GPU acceleration…"
  sudo pacman -Sy --noconfirm nvidia-container-toolkit

  if ! command -v nvidia-ctk &>/dev/null; then
    echo "❌ nvidia-ctk not found after install. Verify nvidia-container-toolkit package." >&2
    exit 1
  fi

  echo "▶ Configuring NVIDIA runtime for Docker…"
  sudo nvidia-ctk runtime configure --runtime=docker >/dev/null
  sudo systemctl restart docker
  echo "✓ NVIDIA container toolkit installed and Docker runtime configured."
  echo "  Restart any running containers to use GPU."
}

# ── Ensure Docker daemon is running ────────────────────────
ensure_docker_daemon() {
  if ! systemctl is-active --quiet docker; then
    echo "▶ Docker daemon is not running. Starting it now…"
    sudo systemctl start docker
    # Poll until the daemon responds (up to 15 s)
    local attempts=0
    while ! systemctl is-active --quiet docker; do
      sleep 0.5
      attempts=$((attempts + 1))
      if [ "$attempts" -ge 30 ]; then
        echo "❌ Docker daemon failed to start within 15 s. Check 'sudo journalctl -u docker'." >&2
        exit 1
      fi
    done
    echo "✓ Docker daemon started."
  fi
}

# ── Prerequisite checks ────────────────────────────────────
check_prereqs() {
  local missing=()
  command -v docker &>/dev/null || missing+=("docker")
  docker compose version &>/dev/null || missing+=("docker-compose plugin")
  if [ ${#missing[@]} -gt 0 ]; then
    echo "❌ Missing: ${missing[*]}" >&2
    if command -v pacman &>/dev/null; then
      echo "   Detected Arch Linux — installing via pacman now…" >&2
      install_docker_arch
      # Re-check after install
      command -v docker &>/dev/null && docker compose version &>/dev/null && return 0
    fi
    echo "   Please install Docker Engine + Compose plugin first." >&2
    exit 1
  fi
  ensure_docker_daemon

  # Verify current user can talk to the daemon without sudo
  if ! docker info &>/dev/null 2>&1; then
    echo "⚠️  Current session lacks permission to access /var/run/docker.sock." >&2
    if id -nG 2>/dev/null | grep -qw docker; then
      echo "   You're in the docker group but the session hasn't picked it up yet." >&2
      echo "   Run 'newgrp docker' or re-login, then retry." >&2
    else
      echo "   Add your user to the docker group:" >&2
      echo "     sudo usermod -aG docker \"$USER\" && newgrp docker" >&2
    fi
    exit 1
  fi
}

# ── GPU check (warn only — backend can fall back to CPU) ───
configure_compose_files() {
  COMPOSE_FILES=(-f "$COMPOSE_BASE_FILE")
  if [ "$GPU_AVAILABLE" -eq 1 ] && [ -f "$COMPOSE_GPU_FILE" ]; then
    COMPOSE_FILES+=( -f "$COMPOSE_GPU_FILE" )
  fi
}

nvidia_runtime_available() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
}

nvidia_hardware_present() {
  command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null
}

configure_nvidia_runtime() {
  if ! command -v nvidia-ctk &>/dev/null; then
    if command -v pacman &>/dev/null; then
      echo "▶ nvidia-ctk not found. Installing nvidia-container-toolkit…"
      if [ "$(id -u)" -eq 0 ]; then
        pacman -Sy --noconfirm nvidia-container-toolkit
      else
        sudo pacman -Sy --noconfirm nvidia-container-toolkit
      fi
    fi
  fi

  if ! command -v nvidia-ctk &>/dev/null; then
    return 1
  fi

  echo "▶ NVIDIA GPU detected. Configuring Docker runtime…"
  if [ "$(id -u)" -eq 0 ]; then
    nvidia-ctk runtime configure --runtime=docker >/dev/null
    systemctl restart docker
  else
    sudo nvidia-ctk runtime configure --runtime=docker >/dev/null
    sudo systemctl restart docker
  fi
}

check_gpu() {
  GPU_AVAILABLE=0

  if nvidia_runtime_available; then
    GPU_AVAILABLE=1
    echo "✓ NVIDIA GPU runtime available"
  elif nvidia_hardware_present; then
    if configure_nvidia_runtime && nvidia_runtime_available; then
      GPU_AVAILABLE=1
      echo "✓ NVIDIA runtime configured and available"
    else
      echo "❌ NVIDIA GPU detected, but Docker runtime is not configured." >&2
      if command -v pacman &>/dev/null; then
        echo "   Run './deploy.sh install-gpu' and retry deployment." >&2
      else
        echo "   Install and configure nvidia-container-toolkit, then retry." >&2
      fi
      exit 1
    fi
  else
    echo "⚠️  NVIDIA hardware/runtime not detected. GPU acceleration will be unavailable."
    if command -v pacman &>/dev/null; then
      echo "   Run './deploy.sh install-gpu' to install nvidia-container-toolkit via pacman."
    else
      echo "   Install nvidia-container-toolkit for 3D reconstruction on GPU."
    fi
  fi

  configure_compose_files
}

# ── Create data directories ────────────────────────────────
setup_dirs() {
  mkdir -p "$SCRIPT_DIR/US_images" "$SCRIPT_DIR/outputs"
  touch "$SCRIPT_DIR/CUS.db" 2>/dev/null || true
}

# ── Remove stale containers that would block startup ───────
# The compose services use fixed container_name values, so a container
# left over from an earlier run (e.g. the folder was renamed, changing the
# Compose project name) collides by name and "docker compose down" in this
# project can't see it. Force-remove any such leftovers before starting.
clean_conflicts() {
  local names=(cus-backend cus-orthanc cus-ohif cus-nifti-viewer)
  local stale=()
  for n in "${names[@]}"; do
    if docker ps -a --format '{{.Names}}' | grep -qx "$n"; then
      stale+=("$n")
    fi
  done
  if [ ${#stale[@]} -gt 0 ]; then
    echo "▶ Removing stale containers: ${stale[*]}"
    docker rm -f "${stale[@]}" >/dev/null 2>&1 || true
  fi
}

# ── Build and start ────────────────────────────────────────
deploy() {
  echo ""
  echo "╔══════════════════════════════════════════════════╗"
  echo "║     CUS Backend — Deploying all services        ║"
  echo "╚══════════════════════════════════════════════════╝"
  echo ""

  cd "$SCRIPT_DIR"
  setup_dirs

  if [ "$GPU_AVAILABLE" -eq 1 ]; then
    echo "▶ Building backend image (GPU-enabled, ~2 min first time)…"
  else
    echo "▶ Building backend image (CPU mode, ~2 min first time)…"
  fi
  compose build backend --quiet 2>/dev/null || compose build backend

  echo "▶ Starting containers…"
  clean_conflicts
  compose up -d --remove-orphans

  verify_services

  echo ""
  show_status
}

# ── Verify services are actually responding ─────────────────
wait_http() {
  local name="$1" url="$2" attempts=0
  while [ "$attempts" -lt 30 ]; do
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$url" 2>/dev/null) || code=000
    case "$code" in
      2*|3*) echo "  ✓ $name"; return 0 ;;
    esac
    sleep 1
    attempts=$((attempts + 1))
  done
  echo "  ✗ $name — not responding at $url (check: ./deploy.sh logs)"
  return 1
}

verify_services() {
  echo ""
  echo "▶ Verifying services…"
  local failed=0
  wait_http "Orthanc REST API   (http://localhost:8042)" "http://localhost:8042/system"        || failed=1
  wait_http "OHIF DICOM Viewer  (http://localhost:3000)" "http://localhost:3000/"              || failed=1
  wait_http "NIfTI 3D Viewer    (http://localhost:3001)" "http://localhost:3001/"              || failed=1
  if timeout 2 bash -c 'echo > /dev/tcp/localhost/8890' 2>/dev/null; then
    echo "  ✓ TCP Image Server   (tcp://localhost:8890)"
  else
    echo "  ✗ TCP Image Server   — port 8890 not accepting connections (check: ./deploy.sh logs backend)"
    failed=1
  fi
  if timeout 2 bash -c 'echo > /dev/tcp/localhost/7556' 2>/dev/null; then
    echo "  ✓ WebSocket Server   (ws://localhost:7556)"
  else
    echo "  ✗ WebSocket Server   — port 7556 not accepting connections (check: ./deploy.sh logs backend)"
    failed=1
  fi
  if [ "$failed" -ne 0 ]; then
    echo ""
    echo "⚠️  Some services failed verification. Recent logs:"
    compose ps
    compose logs --tail=20
  fi
}

# ── Show status + URLs ─────────────────────────────────────
show_status() {
  echo ""
  echo "═══════════════════════════════════════════════════"
  echo "  Service Status"
  echo "═══════════════════════════════════════════════════"
  compose ps 2>/dev/null || true
  echo ""
  echo "═══════════════════════════════════════════════════"
  echo "  Endpoints"
  echo "═══════════════════════════════════════════════════"
  echo "  📡 TCP Image Server   : tcp://localhost:8890"
  echo "  🔌 WebSocket Server  : ws://localhost:7556"
  echo "  🏥 OHIF DICOM Viewer  : http://localhost:3000"
  echo "  🧊 NIfTI 3D Viewer    : http://localhost:3001"
  echo "  🗄️  Orthanc REST API   : http://localhost:8042"
  echo "  📋 Orthanc Explorer   : http://localhost:8042/"
  echo "═══════════════════════════════════════════════════"
  echo ""
}

# ── Stop ───────────────────────────────────────────────────
stop() {
  cd "$SCRIPT_DIR"
  echo "▶ Stopping all services…"
  compose down --remove-orphans
  clean_conflicts
  echo "✓ All containers stopped."
}

# ── Logs ───────────────────────────────────────────────────
logs() {
  cd "$SCRIPT_DIR"
  local service="${1:-}"
  if [ -n "$service" ]; then
    compose logs -f "$service"
  else
    compose logs -f --tail=50
  fi
}

# ── Main ───────────────────────────────────────────────────
case "${1:-deploy}" in
  install-docker)
    install_docker_arch
    ;;
  install-gpu)
    install_nvidia_toolkit_arch
    check_gpu
    ;;
  stop)
    check_prereqs
    stop
    ;;
  logs)
    check_prereqs
    logs "${2:-}"
    ;;
  status)
    check_prereqs
    check_gpu
    show_status
    ;;
  *)
    check_prereqs
    check_gpu
    deploy
    ;;
esac
