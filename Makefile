.PHONY: setup setup-server setup-client install-services test-server run-server run-client logs-server logs-client clean

# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
SERVER_DIR = llm_server
CLIENT_DIR = .
SYSTEMD_DIR = /etc/systemd/system

setup: setup-server setup-client

setup-server:
	python3 -m venv $(SERVER_DIR)/$(VENV_DIR)
	cd $(SERVER_DIR) && ../$(VENV_DIR)/bin/pip install -r requirements.txt
	mkdir -p $(SERVER_DIR)/logs
	mkdir -p $(SERVER_DIR)/models

setup-client:
	python3 -m venv $(VENV_DIR)
	$(PIP) install -r requirements.txt
	mkdir -p logs
	mkdir -p recordings

install-services:
	@echo "Installing systemd services..."
	sudo cp systemd/llm-server.service $(SYSTEMD_DIR)/
	sudo cp systemd/audio-processor.service $(SYSTEMD_DIR)/
	sudo systemctl daemon-reload
	sudo systemctl enable llm-server
	sudo systemctl enable audio-processor
	@echo "Services installed. Use 'sudo systemctl start llm-server' to start."

test-server:
	@echo "Testing LLM server..."
	curl -X POST http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "mistral", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'

run-server:
	cd $(SERVER_DIR) && ../$(VENV_DIR)/bin/python server.py

run-client:
	$(PYTHON) audio_processor.py

logs-server:
	tail -f $(SERVER_DIR)/logs/server.log

logs-client:
	tail -f logs/audio_processor.log

clean:
	rm -rf $(VENV_DIR)
	rm -rf $(SERVER_DIR)/$(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.pyc" -delete
