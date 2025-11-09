import logging

def setup_logging():
    """Configura o sistema de logging"""
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S'
    )