#!/usr/bin/env bash
# =============================================================
#  setup_service.sh
#  Run this ONCE as the "acrossd" user to install and enable
#  the Nyla systemd user service.
#
#  Usage:
#    chmod +x setup_service.sh
#    ./setup_service.sh
# =============================================================

set -euo pipefail

PROJECT_DIR="/home/acrossd/Desktop/nyla"
VENV_PYTHON="$PROJECT_DIR/nyla-env/bin/python"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="nyla.service"

echo ""
echo "============================================================"
echo "  Nyla — Service Installer"
echo "============================================================"

# ── Step 1: Pre-flight checks ─────────────────────────────
echo ""
echo "[1/7] Running pre-flight checks..."

if [ ! -f "$PROJECT_DIR/nyla.py" ]; then
    echo "  ❌ nyla.py not found at $PROJECT_DIR"
    exit 1
fi
echo "  ✅ nyla.py found"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "  ❌ Virtual env not found at $VENV_PYTHON"
    echo "     Create it with: python3 -m venv $PROJECT_DIR/nyla-env"
    exit 1
fi
echo "  ✅ Virtual environment found"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "  ❌ .env file not found at $PROJECT_DIR/.env"
    echo "     Create it and add GROQ_API_KEY=your_key_here"
    exit 1
fi
echo "  ✅ .env file found"

if ! grep -q "GROQ_API_KEY" "$PROJECT_DIR/.env"; then
    echo "  ❌ GROQ_API_KEY not found inside .env"
    exit 1
fi
echo "  ✅ GROQ_API_KEY present in .env"

# ── Step 2: Validate venv python is actually executable ───
# This is the #1 cause of systemd status=203/EXEC:
# a dangling symlink (venv copied from another machine/path)
# or missing execute permission on the binary.
echo ""
echo "[2/7] Validating venv python binary..."

if [ -L "$VENV_PYTHON" ] && [ ! -e "$VENV_PYTHON" ]; then
    echo "  ❌ $VENV_PYTHON is a BROKEN symlink."
    echo "     This happens when a venv is copied/zipped from another"
    echo "     machine or path instead of created fresh. Fix with:"
    echo "       rm -rf $PROJECT_DIR/nyla-env"
    echo "       python3 -m venv $PROJECT_DIR/nyla-env"
    echo "       $PROJECT_DIR/nyla-env/bin/pip install -r requirements.txt"
    exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "  ⚠️  $VENV_PYTHON is not executable — fixing permissions..."
    chmod +x "$VENV_PYTHON"
fi

if ! "$VENV_PYTHON" --version >/dev/null 2>&1; then
    echo "  ❌ $VENV_PYTHON exists but fails to run."
    echo "     Test manually with: $VENV_PYTHON --version"
    exit 1
fi
echo "  ✅ Venv python is executable: $("$VENV_PYTHON" --version 2>&1)"

# ── Step 3: Verify user UID (needed for PulseAudio path) ──
echo ""
echo "[3/7] Checking user UID..."
USER_UID=$(id -u)
if [ "$USER_UID" -ne 1000 ]; then
    echo "  ⚠️  Your UID is $USER_UID (not 1000)."
    echo "     Update PULSE_RUNTIME_PATH in nyla.service to:"
    echo "     Environment=PULSE_RUNTIME_PATH=/run/user/$USER_UID/pulse"
    read -p "  Continue anyway? [y/N] " confirm
    [ "$confirm" = "y" ] || exit 1
else
    echo "  ✅ UID=1000 — PulseAudio path is correct"
fi

# ── Step 4: Install service file ──────────────────────────
echo ""
echo "[4/7] Installing service file..."
mkdir -p "$SERVICE_DIR"
cp "$(dirname "$0")/nyla.service" "$SERVICE_DIR/$SERVICE_NAME"
echo "  ✅ Copied to $SERVICE_DIR/$SERVICE_NAME"

# ── Step 5: Enable linger ─────────────────────────────────
# linger = user session (and its services) start at boot
# WITHOUT requiring a physical login.
echo ""
echo "[5/7] Enabling loginctl linger for user '$USER'..."
sudo loginctl enable-linger "$USER"
echo "  ✅ Linger enabled — service will start at boot without login"

# ── Step 6: Enable and start ──────────────────────────────
echo ""
echo "[6/7] Enabling and starting nyla.service..."
systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user start  "$SERVICE_NAME"

# Brief wait then check
sleep 3
STATUS=$(systemctl --user is-active "$SERVICE_NAME" 2>/dev/null || true)

if [ "$STATUS" = "active" ]; then
    echo "  ✅ Service is RUNNING"
else
    echo "  ⚠️  Service status: $STATUS"
    echo "     Check logs with:"
    echo "       journalctl --user -u $SERVICE_NAME -n 50"
    echo "     If you see status=203/EXEC, re-run this script — step [2/7]"
    echo "     validates the exact failure this error code indicates."
fi

# ── Step 7: Summary ───────────────────────────────────────
echo ""
echo "[7/7] Setup complete."
echo ""
echo "  Useful commands:"
echo "    journalctl --user -u nyla.service -f      # live logs"
echo "    systemctl --user status nyla.service       # quick status"
echo "    systemctl --user restart nyla.service      # manual restart"
echo "    systemctl --user stop nyla.service         # stop"
echo ""
echo "  ⚠️  NEXT STEP: Reboot now and verify the assistant starts automatically."
echo "    sudo reboot"
echo ""
echo "  Only run lockdown.sh AFTER you have confirmed it works post-reboot."
echo "============================================================"
