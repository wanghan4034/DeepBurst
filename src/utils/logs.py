import logging
import colorlog

def get_logger(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s-%(asctime)s-%(levelname)s %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(color_formatter)
    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.addHandler(console_handler)
    
    return logger


if __name__ == '__main__':
    logger = get_logger(logging.DEBUG)
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warning message')
    logger.error('error message')
    logger.critical('critical message')