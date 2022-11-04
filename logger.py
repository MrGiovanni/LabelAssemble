import logging
import colorlog

class Logger:
    def __init__(self, log_path:str)->None:
        """logger

        Args:
            log_path (str): log path
        """
        
        self.log_colors_config = {
            'DEBUG': 'white',  
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
        self.logger = logging.getLogger('logger_name')
        self.console_handler = logging.StreamHandler()
        self.file_handler = logging.FileHandler(filename=log_path, mode='a', encoding='utf8')
        self.logger.setLevel(logging.DEBUG)
        self.console_handler.setLevel(logging.DEBUG)
        self.file_handler.setLevel(logging.INFO)
        self.file_formatter = logging.Formatter(
            fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S'
        )
        self.console_formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S',
            log_colors=self.log_colors_config
        )
        self.console_handler.setFormatter(self.console_formatter)
        self.file_handler.setFormatter(self.file_formatter)
        if not self.logger.handlers:
            self.logger.addHandler(self.console_handler)
            self.logger.addHandler(self.file_handler)
        self.console_handler.close()
        self.file_handler.close()

    def debug(self, message:str)->None:
        self.logger.debug(message)

    def info(self, message:str)->None:
        self.logger.info(message)

    def warning(self, message:str)->None:
        self.logger.warning(message)
    
    def error(self, message:str)->None:
        self.logger.error(message)
    
    def critical(self, message:str)->None:
        self.logger.critical(message)


if __name__ == '__main__':
    logger = Logger('test.log')
    logger.info('Starting')