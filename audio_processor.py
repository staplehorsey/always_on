import logging
from src.audio_processing.orchestrator import AudioOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('audio-processor')

def main():
    processor = AudioOrchestrator()
    try:
        processor.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
