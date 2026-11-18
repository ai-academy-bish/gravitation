# =============================================================================
#  gravitation — Newtonian N-body simulator
#  Run `make` (or `make help`) to list every target.
# =============================================================================

SHELL         := bash
.SHELLFLAGS   := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS     += --no-print-directory

# ---------------------------------------------------------------- variables --
CONFIG   ?= configs/inner_solar_system.yaml
BACKEND  ?= ce
QUALITY  ?= m
BACKENDS ?= ce,ce-gl,gl
DURATION ?=
REPEAT   ?= 1
PRESET   ?= inner
DATE     ?= $(shell date +%Y-%m-%d)
BODIES   ?=
OUT      ?=
EXTRAS   ?= ce,gl
YES      ?=
FORCE    ?=

UV  ?= uv
APP := $(UV) run gravitation

SETUP_FLAGS := --extras $(EXTRAS) $(if $(YES),--yes)
SIM_FLAGS   := $(if $(FORCE),--force)
DUR_FLAGS   := $(if $(DURATION),--duration $(DURATION))

# ---------------------------------------------------------------- colours ----
# Disabled when NO_COLOR is set or when output is not a terminal.
ifeq ($(NO_COLOR)$(shell test -t 2 || echo no),)
  RESET   := \033[0m
  BOLD    := \033[1m
  DIM     := \033[2m
  RED     := \033[31m
  GREEN   := \033[32m
  YELLOW  := \033[33m
  BLUE    := \033[34m
  MAGENTA := \033[35m
  CYAN    := \033[36m
endif

define banner
	@printf "$(BOLD)$(BLUE)==>$(RESET) $(BOLD)%s$(RESET)\n" "$(1)"
endef

# ---------------------------------------------------------------- help -------
##@ Help

.PHONY: help
help: ## Show this help
	@printf "\n  $(BOLD)$(CYAN)✦ gravitation$(RESET)  $(DIM)Newtonian N-body gravity · hand-written integrators · Manim rendering$(RESET)\n"
	@printf "\n  $(BOLD)Usage$(RESET)   make $(CYAN)<target>$(RESET) $(YELLOW)[VAR=value ...]$(RESET)\n"
	@awk 'BEGIN { FS = ":.*?## " } \
		/^##@/ { printf "\n  $(BOLD)$(MAGENTA)%s$(RESET)\n", substr($$0, 5); next } \
		/^[a-zA-Z0-9_.-]+:.*?## / { printf "    $(CYAN)%-13s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\n  $(BOLD)$(MAGENTA)Variables$(RESET)  $(DIM)current value · meaning$(RESET)\n"
	@printf "    $(YELLOW)%-9s$(RESET) $(GREEN)%-34s$(RESET) $(DIM)%s$(RESET)\n" \
		CONFIG   "$(CONFIG)"           "system to simulate (see: make list)" \
		BACKEND  "$(BACKEND)"          "ce (Cairo, CPU) | ce-gl (CE OpenGL) | gl (ManimGL)" \
		QUALITY  "$(QUALITY)"          "l 480p30 | m 720p30 | h 1080p60 | k 2160p60" \
		DURATION "$(or $(DURATION),from config)" "video length in seconds" \
		BACKENDS "$(BACKENDS)"         "backends compared by make bench" \
		REPEAT   "$(REPEAT)"           "renders per backend in make bench" \
		PRESET   "$(PRESET)"           "ephemeris preset: inner | outer | full" \
		DATE     "$(DATE)"             "ephemeris epoch (YYYY-MM-DD)" \
		BODIES   "$(or $(BODIES),from preset)" "ephemeris bodies, e.g. sun,earth,moon" \
		EXTRAS   "$(EXTRAS)"           "renderers installed by setup: ce,gl | ce | gl | none" \
		YES      "$(or $(YES),-)"      "YES=1 answers yes to every setup prompt" \
		FORCE    "$(or $(FORCE),-)"    "FORCE=1 re-runs the simulation even if cached"
	@printf "\n  $(BOLD)$(MAGENTA)Examples$(RESET)\n"
	@printf "    $(DIM)\$$$(RESET) make setup\n"
	@printf "    $(DIM)\$$$(RESET) make render CONFIG=configs/figure_eight.yaml QUALITY=h\n"
	@printf "    $(DIM)\$$$(RESET) make ephemeris PRESET=full DATE=1969-07-20\n"
	@printf "    $(DIM)\$$$(RESET) make bench QUALITY=l DURATION=5\n\n"

# ---------------------------------------------------------------- setup ------
##@ Setup

.PHONY: setup sysdeps install doctor
setup: ## Everything: system packages (brew/apt/dnf/pacman), uv, .venv, Python deps
	@bash scripts/setup.sh $(SETUP_FLAGS)

sysdeps: ## Install only the missing system packages
	@bash scripts/setup.sh --system-only $(SETUP_FLAGS)

install: ## Create .venv and install Python deps with uv (no system packages)
	@bash scripts/setup.sh --python-only $(SETUP_FLAGS)

doctor: ## Report what is installed and what is missing
	@bash scripts/setup.sh --check --extras $(EXTRAS)

# ---------------------------------------------------------------- simulation -
##@ Simulation

.PHONY: list simulate preview compare ephemeris colors
list: ## List the available configs
	@$(APP) list

simulate: ## Integrate CONFIG and print energy/momentum diagnostics
	$(call banner,Simulating $(CONFIG))
	@$(APP) simulate $(CONFIG) $(SIM_FLAGS)

preview: ## Quick matplotlib preview (orbits + energy error) as PNG
	$(call banner,Preview of $(CONFIG))
	@$(APP) preview $(CONFIG) $(SIM_FLAGS)

compare: ## Run every integrator on CONFIG and compare their accuracy
	$(call banner,Comparing integrators on $(CONFIG))
	@$(APP) compare $(CONFIG)

ephemeris: ## Fetch real initial conditions from NASA JPL Horizons
	$(call banner,JPL Horizons: $(PRESET) on $(DATE))
	@$(APP) ephemeris --preset $(PRESET) --date $(DATE) $(if $(BODIES),--bodies $(BODIES)) $(if $(OUT),--out $(OUT))

colors: ## Show the named colours accepted in configs
	@$(APP) colors

# ---------------------------------------------------------------- rendering --
##@ Rendering

.PHONY: render render-all bench
render: ## Render CONFIG to video with BACKEND at QUALITY
	$(call banner,Rendering $(CONFIG) with $(BACKEND) at quality $(QUALITY))
	@$(APP) render $(CONFIG) --backend $(BACKEND) --quality $(QUALITY) $(DUR_FLAGS) $(SIM_FLAGS)

render-all: ## Render CONFIG with every backend in BACKENDS
	@for b in $$(echo "$(BACKENDS)" | tr ',' ' '); do \
		$(MAKE) render BACKEND=$$b CONFIG=$(CONFIG) QUALITY=$(QUALITY) DURATION=$(DURATION); \
	done

bench: ## Benchmark the rendering backends and write a Markdown report
	$(call banner,Benchmarking $(BACKENDS) on $(CONFIG))
	@$(APP) bench $(CONFIG) --backends $(BACKENDS) --quality $(QUALITY) --repeat $(REPEAT) $(DUR_FLAGS)

# ---------------------------------------------------------------- dev --------
##@ Development

.PHONY: test lint format clean distclean
test: ## Run the test suite
	@$(UV) run pytest

lint: ## Check style and common bugs with ruff
	@$(UV) run ruff check src tests

format: ## Auto-format and fix imports with ruff
	@$(UV) run ruff check --fix src tests
	@$(UV) run ruff format src tests

clean: ## Remove caches and rendered output (keeps .venv)
	$(call banner,Cleaning)
	@rm -rf output media .pytest_cache .ruff_cache
	@find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
	@printf "    $(GREEN)✔$(RESET) done\n"

distclean: clean ## Also remove the virtual environment
	@rm -rf .venv
	@printf "    $(GREEN)✔$(RESET) .venv removed\n"
