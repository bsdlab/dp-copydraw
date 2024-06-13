from fire import Fire

from copydraw.utils.logging import logger
from copydraw.main import init_paradigm

from dareplane_utils.default_server.server import DefaultServer


def main(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 10):
    logger.setLevel(loglevel)

    copydraw = init_paradigm()
    pcommand_map = {
        'START_BLOCK': copydraw.exec_block,
        # 'STOP': lambda: 1,     # functionality is implemented in the main loop, here for book keeping
        # 'CLOSE': lambda: 1,
    }

    logger.debug('Initializing server')
    server = DefaultServer(
        port, ip=ip, pcommand_map=pcommand_map, name="copydraw_server"
    )

    # initialize to start the socket
    server.init_server()
    # start processing of the server

    logger.debug('starting to listen on server')
    server.start_listening()
    logger.debug('stopped to listen on server')

    return 0


if __name__ == "__main__":
    Fire(main)
