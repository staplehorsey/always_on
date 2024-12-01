# Makefile for always-on audio processing system

.PHONY: all clean install-deps install-services start stop restart status logs test

# Virtual environment
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Service names
AUDIO_SERVICE := audio-processor.service

all: $(VENV) install-deps

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install-deps: $(VENV)
	$(PIP) install -r requirements.txt

clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf *.pyc

# Systemd service management
install-services:
	sudo cp systemd/$(AUDIO_SERVICE) /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable $(AUDIO_SERVICE)

start:
	sudo systemctl start $(AUDIO_SERVICE)

stop:
	sudo systemctl stop $(AUDIO_SERVICE)

restart:
	sudo systemctl restart $(AUDIO_SERVICE)

status:
	sudo systemctl status $(AUDIO_SERVICE)

logs:
	sudo journalctl -u $(AUDIO_SERVICE) -f

test:
	$(PYTHON) -m pytest

# Development targets
run-audio:
	$(PYTHON) audio_processor.py

.DEFAULT_GOAL := all
