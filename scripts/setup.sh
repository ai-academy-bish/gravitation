#!/usr/bin/env bash
# =============================================================================
#  gravitation — one-shot environment setup
#
#  1. detects the OS and package manager (brew, apt, dnf, pacman, zypper)
#  2. installs only the system packages that are missing
#  3. installs uv if needed
#  4. creates .venv and installs the Python dependencies with `uv sync`
#
#  Written for bash 3.2 (the macOS default): no associative arrays.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- options ----
MODE="all"          # all | system | python | check
EXTRAS="ce,gl"
ASSUME_YES=0
DRY_RUN=0
HEADLESS=1

usage() {
  cat <<EOF
Usage: scripts/setup.sh [options]

Sets up everything needed to run and render gravitation simulations.

Options:
  --check           Report what is installed and what is missing; change nothing
  --system-only     Install missing system packages only
  --python-only     Skip system packages; create .venv and install Python deps
  --extras LIST     Renderers to install: ce,gl (default) | ce | gl | none
  --no-headless     On Linux without a display, do not install Xvfb
  -y, --yes         Do not ask for confirmation (needed when not on a terminal)
  -n, --dry-run     Print the commands instead of running them
  -h, --help        Show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE="check" ;;
    --system-only) MODE="system" ;;
    --python-only) MODE="python" ;;
    --extras) shift; EXTRAS="${1:-}" ;;
    --extras=*) EXTRAS="${1#*=}" ;;
    --no-headless) HEADLESS=0 ;;
    -y|--yes) ASSUME_YES=1 ;;
    -n|--dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case ",$EXTRAS," in
  *,none,*) EXTRAS="" ;;
esac
want_extra() { case ",$EXTRAS," in *,"$1",*) return 0 ;; *) return 1 ;; esac; }

# ---------------------------------------------------------------- output -----
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B=$'\033[1m'; D=$'\033[2m'; R=$'\033[0m'
  RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'; MAG=$'\033[35m'; CYN=$'\033[36m'
else
  B=""; D=""; R=""; RED=""; GRN=""; YLW=""; BLU=""; MAG=""; CYN=""
fi

step()  { printf '\n%s==>%s %s%s%s\n' "$BLU$B" "$R" "$B" "$*" "$R"; }
info()  { printf '    %s\n' "$*"; }
ok()    { printf '    %s✔%s %s\n' "$GRN" "$R" "$*"; }
miss()  { printf '    %s✘%s %s\n' "$RED" "$R" "$*"; }
warn()  { printf '    %s!%s %s\n' "$YLW" "$R" "$*"; }
die()   { printf '\n%serror:%s %s\n' "$RED$B" "$R" "$*" >&2; exit 1; }

run() {
  printf '    %s$ %s%s\n' "$D" "$*" "$R"
  if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

confirm() {
  [ "$ASSUME_YES" -eq 1 ] && return 0
  if [ ! -t 0 ]; then
    die "not running on a terminal; re-run with --yes (or: make setup YES=1)"
  fi
  local answer
  printf '    %s?%s %s [Y/n] ' "$MAG" "$R" "$1"
  read -r answer || answer="n"
  case "$answer" in ""|y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

# ---------------------------------------------------------------- platform ---
OS="$(uname -s)"
PM=""
case "$OS" in
  Darwin) command -v brew >/dev/null 2>&1 && PM="brew" ;;
  Linux)
    for candidate in apt-get dnf pacman zypper; do
      if command -v "$candidate" >/dev/null 2>&1; then PM="$candidate"; break; fi
    done ;;
  MINGW*|MSYS*|CYGWIN*)
    die "native Windows is not supported by this script; please use WSL2 (Ubuntu) and run it there" ;;
  *) die "unsupported OS: $OS" ;;
esac

SUDO=""
if [ "$OS" = "Linux" ] && [ "$(id -u)" -ne 0 ]; then
  command -v sudo >/dev/null 2>&1 && SUDO="sudo"
fi

HAS_DISPLAY=1
if [ "$OS" = "Linux" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  HAS_DISPLAY=0
fi

# ---------------------------------------------------------------- requirements
# Each requirement has three functions:
#   applies_<req>  — is it needed on this platform with the chosen extras?
#   check_<req>    — is it already present?
#   pkgs_<req>     — package names for the detected package manager
REQUIREMENTS="compiler pkgconfig cairo pango ffmpeg vulkan opengl xvfb"

has_pkgconfig() { command -v pkg-config >/dev/null 2>&1 || command -v pkgconf >/dev/null 2>&1; }
pc() { if command -v pkg-config >/dev/null 2>&1; then pkg-config "$@"; else pkgconf "$@"; fi; }
has_lib() { ldconfig -p 2>/dev/null | grep -q "$1" || ls /usr/lib*/"$1"* /usr/lib/*-linux-gnu/"$1"* >/dev/null 2>&1; }

describe_compiler()  { echo "C compiler (builds pycairo for Manim CE)"; }
applies_compiler()   { want_extra ce; }
check_compiler() {
  if [ "$OS" = "Darwin" ]; then xcode-select -p >/dev/null 2>&1; else command -v cc >/dev/null 2>&1; fi
}
pkgs_compiler() {
  case "$PM" in
    apt-get) echo "build-essential" ;; dnf) echo "gcc" ;; pacman) echo "base-devel" ;; zypper) echo "gcc" ;;
  esac
}

describe_pkgconfig() { echo "pkg-config (locates cairo/pango)"; }
applies_pkgconfig()  { want_extra ce; }
check_pkgconfig()    { has_pkgconfig; }
pkgs_pkgconfig() {
  case "$PM" in
    brew) echo "pkgconf" ;; apt-get) echo "pkg-config" ;; dnf) echo "pkgconf-pkg-config" ;;
    pacman) echo "pkgconf" ;; zypper) echo "pkg-config" ;;
  esac
}

describe_cairo() { echo "cairo (2-D graphics library behind Manim CE)"; }
applies_cairo()  { want_extra ce; }
check_cairo()    { has_pkgconfig && pc --exists cairo; }
pkgs_cairo() {
  case "$PM" in
    brew) echo "cairo" ;; apt-get) echo "libcairo2-dev" ;; dnf) echo "cairo-devel" ;;
    pacman) echo "cairo" ;; zypper) echo "cairo-devel" ;;
  esac
}

# manimpango ships macOS wheels but no Linux x86_64 wheels, so Linux builds it.
describe_pango() { echo "pango (text layout; manimpango builds against it on Linux)"; }
applies_pango()  { [ "$OS" = "Linux" ] && { want_extra ce || want_extra gl; }; }
check_pango()    { has_pkgconfig && pc --exists pangocairo; }
pkgs_pango() {
  case "$PM" in
    apt-get) echo "libpango1.0-dev" ;; dnf) echo "pango-devel" ;; pacman) echo "pango" ;; zypper) echo "pango-devel" ;;
  esac
}

# Manim CE encodes video through PyAV (bundled FFmpeg); ManimGL pipes frames to the ffmpeg binary.
describe_ffmpeg() { echo "ffmpeg (video encoder used by ManimGL)"; }
applies_ffmpeg()  { want_extra gl; }
check_ffmpeg()    { command -v ffmpeg >/dev/null 2>&1; }
pkgs_ffmpeg() {
  case "$PM" in
    brew|apt-get|pacman|zypper) echo "ffmpeg" ;; dnf) echo "ffmpeg-free" ;;
  esac
}

# ManimGL renders through wgpu: Metal on macOS (nothing to install), Vulkan on
# Linux. Mesa's Vulkan drivers include lavapipe, a software renderer, so even
# a headless server without a GPU can render.
describe_vulkan() { echo "Vulkan loader + Mesa drivers (ManimGL renders through wgpu)"; }
applies_vulkan()  { [ "$OS" = "Linux" ] && want_extra gl; }
check_vulkan() {
  has_lib libvulkan.so.1 && ls /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json >/dev/null 2>&1
}
pkgs_vulkan() {
  case "$PM" in
    apt-get) echo "libvulkan1 mesa-vulkan-drivers" ;; dnf) echo "vulkan-loader mesa-vulkan-drivers" ;;
    pacman) echo "vulkan-icd-loader vulkan-swrast" ;; zypper) echo "libvulkan1 libvulkan_lvp" ;;
  esac
}

# Manim CE's optional OpenGL renderer (backend ce-gl) needs libGL and, without
# a display, an X server to create its context: Xvfb provides a virtual one.
describe_opengl() { echo "OpenGL/EGL runtime (Manim CE's OpenGL renderer)"; }
applies_opengl()  { [ "$OS" = "Linux" ] && want_extra ce; }
check_opengl()    { has_lib libGL.so.1 && has_lib libEGL.so.1; }
pkgs_opengl() {
  case "$PM" in
    apt-get) echo "libgl1 libegl1" ;; dnf) echo "mesa-libGL mesa-libEGL" ;; pacman) echo "mesa" ;;
    zypper) echo "Mesa-libGL1 Mesa-libEGL1" ;;
  esac
}

describe_xvfb() { echo "Xvfb (virtual display for Manim CE's OpenGL renderer on headless Linux)"; }
applies_xvfb()  { [ "$OS" = "Linux" ] && [ "$HAS_DISPLAY" -eq 0 ] && [ "$HEADLESS" -eq 1 ] && want_extra ce; }
check_xvfb()    { command -v xvfb-run >/dev/null 2>&1; }
pkgs_xvfb() {
  case "$PM" in
    apt-get) echo "xvfb" ;; dnf) echo "xorg-x11-server-Xvfb" ;; pacman) echo "xorg-server-xvfb" ;;
    zypper) echo "xvfb-run" ;;
  esac
}

MISSING_PKGS=""
NEED_XCODE=0

scan_system() {
  step "System packages  ${D}($OS, package manager: ${PM:-none})${R}"
  local req pkgs
  for req in $REQUIREMENTS; do
    "applies_$req" || continue
    if "check_$req"; then
      ok "$("describe_$req")"
    else
      if [ "$OS" = "Darwin" ] && [ "$req" = "compiler" ]; then
        NEED_XCODE=1
        miss "$("describe_$req")  ${D}→ Xcode Command Line Tools${R}"
        continue
      fi
      pkgs="$("pkgs_$req" || true)"
      miss "$("describe_$req")  ${D}→ ${pkgs:-no package known for ${PM:-this system}}${R}"
      [ -n "$pkgs" ] && MISSING_PKGS="$MISSING_PKGS $pkgs"
    fi
  done
  MISSING_PKGS="$(echo "$MISSING_PKGS" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ')"
  MISSING_PKGS="${MISSING_PKGS% }"
}

install_system() {
  if [ "$NEED_XCODE" -eq 1 ]; then
    warn "Xcode Command Line Tools are required to compile pycairo"
    if confirm "Run 'xcode-select --install' (opens a macOS dialog)?"; then
      run xcode-select --install || true
      die "finish the Xcode Command Line Tools installation, then run setup again"
    fi
  fi
  if [ -z "$MISSING_PKGS" ]; then
    ok "nothing to install"
    return
  fi
  if [ -z "$PM" ]; then
    if [ "$OS" = "Darwin" ]; then
      die "Homebrew is required: see https://brew.sh, then run setup again"
    fi
    die "no supported package manager found; install manually: $MISSING_PKGS"
  fi
  confirm "Install with $PM: $MISSING_PKGS ?" || die "aborted by user"
  # shellcheck disable=SC2086 — package lists are intentionally word-split
  case "$PM" in
    brew)    run brew install $MISSING_PKGS ;;
    apt-get) run $SUDO apt-get update -qq
             run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $MISSING_PKGS ;;
    dnf)     run $SUDO dnf install -y $MISSING_PKGS ;;
    pacman)  run $SUDO pacman -S --needed --noconfirm $MISSING_PKGS ;;
    zypper)  run $SUDO zypper --non-interactive install $MISSING_PKGS ;;
  esac
}

# ---------------------------------------------------------------- uv & Python
ensure_uv() {
  step "uv  ${D}(Python package and environment manager)${R}"
  if command -v uv >/dev/null 2>&1; then
    ok "$(uv --version)"
    return
  fi
  miss "uv not found"
  [ "$MODE" = "check" ] && return
  confirm "Install uv?" || die "uv is required"
  if [ "$PM" = "brew" ]; then
    run brew install uv
  elif command -v curl >/dev/null 2>&1; then
    printf '    %s$ curl -LsSf https://astral.sh/uv/install.sh | sh%s\n' "$D" "$R"
    [ "$DRY_RUN" -eq 1 ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    printf '    %s$ wget -qO- https://astral.sh/uv/install.sh | sh%s\n' "$D" "$R"
    [ "$DRY_RUN" -eq 1 ] || wget -qO- https://astral.sh/uv/install.sh | sh
  else
    die "need curl or wget to install uv"
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  [ "$DRY_RUN" -eq 1 ] || command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; open a new shell and re-run"
}

sync_python() {
  step "Python environment  ${D}(.venv, extras: ${EXTRAS:-none})${R}"
  local args="sync"
  local extra
  for extra in $(echo "$EXTRAS" | tr ',' ' '); do args="$args --extra $extra"; done
  # shellcheck disable=SC2086
  run uv $args
  if [ "$DRY_RUN" -eq 0 ]; then
    check_python
  fi
}

check_python() {
  if [ ! -x "$ROOT/.venv/bin/python" ]; then
    miss ".venv not created yet"
    return
  fi
  ok ".venv  ${D}($("$ROOT/.venv/bin/python" --version 2>&1))${R}"
  local spec module label version
  for spec in "gravitation:gravitation" "manim:Manim CE" "manimlib:ManimGL"; do
    module="${spec%%:*}"; label="${spec#*:}"
    version="$("$ROOT/.venv/bin/python" - "$module" 2>/dev/null <<'PY' || true
import importlib.metadata as md, importlib.util, sys
name = sys.argv[1]
if importlib.util.find_spec(name) is None:
    sys.exit(1)
dist = {"manimlib": "manimgl"}.get(name, name)
print(md.version(dist))
PY
)"
    if [ -n "$version" ]; then ok "$label $version"; else miss "$label not installed"; fi
  done
}

# ---------------------------------------------------------------- main -------
printf '%s✦ gravitation setup%s  %s%s%s\n' "$CYN$B" "$R" "$D" "$ROOT" "$R"
[ "$DRY_RUN" -eq 1 ] && warn "dry run: commands are printed, not executed"

case "$MODE" in
  check)
    scan_system
    ensure_uv
    step "Python environment"
    check_python
    if [ -n "$MISSING_PKGS" ] || [ "$NEED_XCODE" -eq 1 ]; then
      printf '\n%sSome system packages are missing:%s run %smake setup%s\n' "$YLW" "$R" "$B" "$R"
    fi
    ;;
  system)
    scan_system
    install_system
    ;;
  python)
    ensure_uv
    sync_python
    ;;
  all)
    scan_system
    install_system
    ensure_uv
    sync_python
    ;;
esac

if [ "$MODE" = "all" ] || [ "$MODE" = "python" ]; then
  printf '\n%s✔ Ready.%s Try: %smake simulate%s, %smake preview%s, %smake render%s  (or %smake help%s)\n' \
    "$GRN$B" "$R" "$CYN" "$R" "$CYN" "$R" "$CYN" "$R" "$CYN" "$R"
fi
if [ "$OS" = "Linux" ] && [ "$HAS_DISPLAY" -eq 0 ]; then
  info "${D}No display: ManimGL renders off-screen through Vulkan; the ce-gl backend runs under xvfb-run.${R}"
fi
