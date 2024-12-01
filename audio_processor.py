import os
import sys
import logging
import platform
import configparser
from pathlib import Path
from src.audio_processing.orchestrator import AudioOrchestrator

def get_platform_paths():
    """Get platform-specific default paths"""
    system = platform.system()
    home = str(Path.home())
    
    if system == "Darwin":  # macOS
        return {
            'log_dir': os.path.join(home, 'Library/Logs/always_on'),
            'config_dir': os.path.join(home, 'Library/Application Support/always_on'),
            'recordings_dir': os.path.join(home, 'Library/Application Support/always_on/recordings')
        }
    else:  # Linux and others
        return {
            'log_dir': '/var/log/always_on',
            'config_dir': '/etc/always_on',
            'recordings_dir': '/var/lib/always_on/recordings'
        }

def load_config():
    """Load configuration from file"""
    config = configparser.ConfigParser()
    platform_paths = get_platform_paths()
    
    # Try loading from different locations
    config_paths = [
        os.path.join(platform_paths['config_dir'], 'always_on.conf'),  # System config
        os.path.join(os.path.dirname(__file__), 'config/always_on.conf')  # Local config
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                # Add a default section for ConfigParser
                config_str = '[DEFAULT]\n' + f.read()
            config.read_string(config_str)
            
            # Override paths with platform-specific ones if not explicitly set
            config_dict = dict(config['DEFAULT'])
            if 'LOG_DIR' not in config_dict:
                config_dict['LOG_DIR'] = platform_paths['log_dir']
            if 'RECORDINGS_DIR' not in config_dict:
                config_dict['RECORDINGS_DIR'] = platform_paths['recordings_dir']
                
            return config_dict
    
    # If no config file found, return default values with platform-specific paths
    return {
        'SERVER_HOST': 'staple.local',
        'SERVER_PORT': '12345',
        'ENABLE_MONITORING': 'false',
        'LOG_LEVEL': 'INFO',
        'LOG_DIR': platform_paths['log_dir'],
        'RECORDINGS_DIR': platform_paths['recordings_dir']
    }

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = config.get('LOG_DIR')
    log_level = getattr(logging, config.get('LOG_LEVEL', 'DEBUG'))
    
    # Create log directory if it doesn't exist
    try:
        os.makedirs(log_dir, exist_ok=True)
    except PermissionError:
        # Fall back to user's home directory if we can't write to system directories
        home = str(Path.home())
        log_dir = os.path.join(home, '.local/share/always_on/logs')
        os.makedirs(log_dir, exist_ok=True)
        config['LOG_DIR'] = log_dir
    
    # Remove all existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'always_on.log'))
        ]
    )
    
    return logging.getLogger('audio-processor')

def main():
    # Load configuration first
    try:
        config = load_config()
        
        # Setup logging once
        logger = setup_logging(config)
        
        # Create necessary directories
        os.makedirs(config['RECORDINGS_DIR'], exist_ok=True)
        
        # Create processor with configuration
        processor = AudioOrchestrator(
            server_host=config.get('SERVER_HOST', 'staple.local'),
            server_port=int(config.get('SERVER_PORT', '12345')),
            enable_monitoring=config.get('ENABLE_MONITORING', 'true').lower() == 'true',
            recordings_dir=config.get('RECORDINGS_DIR')
        )
        
        processor.run()
        
    except KeyboardInterrupt:
        if 'logger' in locals():
            logger.info("Shutting down...")
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Fatal error: {e}")
        else:
            print(f"Fatal error before logging was initialized: {e}")
        raise

if __name__ == "__main__":
    main()
