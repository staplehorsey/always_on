# Makefile for always-on audio processing system

.PHONY: all clean install install-deps install-service start stop restart status logs test uninstall configure

# Configuration
include config/always_on.conf

# Virtual environment
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Detect platform
PLATFORM := $(shell uname)

# Platform-specific settings
ifeq ($(PLATFORM),Darwin)
    # macOS settings
    SERVICE_NAME := com.always_on.audio
    START_CMD := sudo launchctl load -w /Library/LaunchDaemons/$(SERVICE_NAME).plist
    STOP_CMD := sudo launchctl unload -w /Library/LaunchDaemons/$(SERVICE_NAME).plist
    STATUS_CMD := launchctl list | grep $(SERVICE_NAME) || echo "Service not running"
    LOGS_CMD := tail -f /Library/Logs/always_on/always_on.log
else
    # Linux settings
    SERVICE_NAME := always_on@$(SERVICE_USER)
    START_CMD := sudo systemctl start $(SERVICE_NAME)
    STOP_CMD := sudo systemctl stop $(SERVICE_NAME)
    STATUS_CMD := sudo systemctl status $(SERVICE_NAME)
    LOGS_CMD := sudo journalctl -u $(SERVICE_NAME) -f
endif

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

# Complete installation
install: install-deps
	@echo "Installing Always On service..."
	sudo bash install.sh

start:
	$(START_CMD)

stop:
	$(STOP_CMD)

restart: stop start

status:
	$(STATUS_CMD)

logs:
	$(LOGS_CMD)

uninstall:
	@echo "Uninstalling Always On service..."
ifeq ($(PLATFORM),Darwin)
	$(STOP_CMD) || true
	sudo rm -f /Library/LaunchDaemons/$(SERVICE_NAME).plist
	sudo rm -rf "/Library/Application Support/always_on"
	sudo rm -rf "/Library/Logs/always_on"
else
	sudo systemctl stop $(SERVICE_NAME) || true
	sudo systemctl disable $(SERVICE_NAME) || true
	sudo rm -f /etc/systemd/system/always_on@.service
	sudo rm -rf /opt/always_on
	sudo rm -rf /etc/always_on
	sudo userdel -r $(SERVICE_USER) || true
	sudo groupdel $(SERVICE_GROUP) || true
	sudo systemctl daemon-reload
endif

configure:
	@echo "Current configuration:"
	@echo "Platform: $(PLATFORM)"
ifeq ($(PLATFORM),Darwin)
	@echo "Recordings directory: /Library/Application Support/always_on/recordings"
	@echo "Log directory: /Library/Logs/always_on"
else
	@echo "Recordings directory: $(RECORDINGS_DIR)"
	@echo "Log directory: $(LOG_DIR)"
	@echo "Service user: $(SERVICE_USER)"
endif
	@echo "Server host: $(SERVER_HOST)"
	@echo "Server port: $(SERVER_PORT)"
	@echo ""
	@echo "To modify configuration, edit config/always_on.conf"

test:
	$(PYTHON) -m pytest

# Development targets
run:
	$(PYTHON) audio_processor.py

.DEFAULT_GOAL := all
