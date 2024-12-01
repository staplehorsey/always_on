#!/bin/bash
set -e

# Source configuration
source config/always_on.conf

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@"
}

# Detect platform
PLATFORM=$(uname)

if [ "$PLATFORM" = "Darwin" ]; then
    # macOS installation
    log "Installing for macOS..."
    
    # Check if running with sudo
    if [ "$EUID" -ne 0 ]; then
        echo "Please run with sudo for system-wide installation"
        exit 1
    fi
    
    # Create application directories
    APP_SUPPORT_DIR="/Library/Application Support/always_on"
    LOG_DIR="/Library/Logs/always_on"
    
    mkdir -p "$APP_SUPPORT_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$APP_SUPPORT_DIR/recordings"
    
    # Install Python dependencies
    python3 -m pip install -r requirements.txt
    
    # Copy files
    cp -r src "$APP_SUPPORT_DIR/"
    cp audio_processor.py "$APP_SUPPORT_DIR/"
    mkdir -p "$APP_SUPPORT_DIR/config"
    cp config/always_on.conf "$APP_SUPPORT_DIR/config/"
    
    # Create launch daemon
    cat > /Library/LaunchDaemons/com.always_on.audio.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.always_on.audio</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>$APP_SUPPORT_DIR/audio_processor.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$APP_SUPPORT_DIR</string>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/always_on.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/always_on.error.log</string>
</dict>
</plist>
EOF
    
    # Set permissions
    chown -R root:wheel "$APP_SUPPORT_DIR"
    chown -R root:wheel "$LOG_DIR"
    chmod 755 "$APP_SUPPORT_DIR"
    chmod 755 "$LOG_DIR"
    chmod 644 /Library/LaunchDaemons/com.always_on.audio.plist
    
    # Load launch daemon
    launchctl load -w /Library/LaunchDaemons/com.always_on.audio.plist
    
    log "macOS installation complete!"
    log "Logs are in: $LOG_DIR"
    log "Recordings will be saved in: $APP_SUPPORT_DIR/recordings"
    
else
    # Linux installation
    log "Installing for Linux..."
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        echo "Please run as root"
        exit 1
    fi
    
    # Install system dependencies
    log "Installing system dependencies..."
    if [ -f /etc/debian_version ]; then
        apt-get update
        apt-get install -y $SYSTEM_DEPENDENCIES
    elif [ -f /etc/redhat-release ]; then
        yum install -y $SYSTEM_DEPENDENCIES
    else
        log "Unsupported Linux distribution. Please install dependencies manually."
    fi
    
    # Create service user and group
    log "Creating service user and group..."
    if ! getent group "$SERVICE_GROUP" >/dev/null; then
        groupadd -r "$SERVICE_GROUP"
    fi
    if ! getent passwd "$SERVICE_USER" >/dev/null; then
        useradd -r -g "$SERVICE_GROUP" -d /opt/always_on -s /sbin/nologin "$SERVICE_USER"
    fi
    
    # Create necessary directories
    log "Creating directories..."
    mkdir -p /opt/always_on
    mkdir -p "$RECORDINGS_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p /etc/always_on
    
    # Set up Python virtual environment
    log "Setting up Python virtual environment..."
    python3 -m venv /opt/always_on/venv
    /opt/always_on/venv/bin/pip install --upgrade pip
    /opt/always_on/venv/bin/pip install -r requirements.txt
    
    # Copy files
    log "Copying files..."
    cp -r src /opt/always_on/
    cp audio_processor.py /opt/always_on/
    cp config/always_on.conf /etc/always_on/
    cp config/always_on.service /etc/systemd/system/always_on@.service
    
    # Set permissions
    log "Setting permissions..."
    chown -R "$SERVICE_USER:$SERVICE_GROUP" /opt/always_on
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$RECORDINGS_DIR"
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$LOG_DIR"
    chmod 755 /opt/always_on
    chmod 755 "$RECORDINGS_DIR"
    chmod 755 "$LOG_DIR"
    
    # Configure log rotation
    log "Configuring log rotation..."
    cat > /etc/logrotate.d/always_on << EOF
$LOG_DIR/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 640 $SERVICE_USER $SERVICE_GROUP
    sharedscripts
    postrotate
        systemctl restart always_on@$SERVICE_USER
    endscript
}
EOF
    
    # Reload systemd
    log "Reloading systemd..."
    systemctl daemon-reload
    
    # Enable and start service
    log "Enabling and starting service..."
    systemctl enable "always_on@$SERVICE_USER"
    systemctl start "always_on@$SERVICE_USER"
    
    log "Linux installation complete!"
fi

log "Installation complete! Check logs for status."
