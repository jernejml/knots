# Build pipeline for the wood-knot dataset preprocessing.
#
# Targets:
#   make all     run the chain: analyze_dataset -> board_features
#   make stats   run dataset_stats.py (diagnostic; no downstream consumers)
#   make clean   remove out/analysis/ outputs
#
# Notes:
# - Targets are tracked by output-file timestamps. Editing a script
#   causes its target to rebuild on the next 'make'.
# - The pipeline does NOT depend on data/ contents, only on script
#   timestamps. If you swap data/ but leave scripts/ unchanged, run
#   'make -B all' or 'make clean all' to force a rebuild.
# - CLI flag changes (e.g. --seed) are NOT seen by make. For iteration
#   on script flags, invoke that script directly instead of via make.

.PHONY: all stats clean

PY       := python3
SCRIPTS  := scripts
ANALYSIS := out/analysis

# analyze_dataset.py produces both frames.json and annotations.json in
# one invocation. Grouped-target syntax ('&:') tells make a single run
# builds all of them (GNU make >= 4.3).
$(ANALYSIS)/frames.json $(ANALYSIS)/annotations.json &: $(SCRIPTS)/analyze_dataset.py
	$(PY) $<

$(ANALYSIS)/board_features.json: $(ANALYSIS)/frames.json $(ANALYSIS)/annotations.json $(SCRIPTS)/board_features.py
	$(PY) $(SCRIPTS)/board_features.py

$(ANALYSIS)/splits.csv: $(ANALYSIS)/board_features.json $(SCRIPTS)/make_splits.py
	$(PY) $(SCRIPTS)/make_splits.py

all: $(ANALYSIS)/splits.csv

stats:
	$(PY) $(SCRIPTS)/dataset_stats.py

clean:
	rm -f $(ANALYSIS)/*.csv $(ANALYSIS)/*.json
