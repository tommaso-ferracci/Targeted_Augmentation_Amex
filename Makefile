# Define variables
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python3
PIP := $(VENV_DIR)/bin/pip
JUPYTER := $(VENV_DIR)/bin/jupyter
REQUIREMENTS := requirements.txt
NOTEBOOK := analyses/notebook.ipynb

# Default target: create virtual environment, install dependencies, and run notebook
all: venv install execute

# Target to create a virtual environment
venv:
	python3 -m venv $(VENV_DIR)

# Target to install requirements
install: venv
	$(PIP) install -r $(REQUIREMENTS)

# Target to execute the Jupyter notebook
execute: install
	$(JUPYTER) nbconvert --to notebook --execute $(NOTEBOOK) --output $(NOTEBOOK)

# Clean target to remove the virtual environment and the generated notebook
clean:
	rm -rf $(VENV_DIR) $(OUTPUT_NOTEBOOK)

