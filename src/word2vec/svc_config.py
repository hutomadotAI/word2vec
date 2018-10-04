import os


class SvcConfig(object):

    __instance = None

    def __init__(self):
        self._vectors_file = {
            'en': os.environ.get(
                'W2V_VECTOR_FILE_EN', '/datasets/GoogleNews-vectors-negative300.bin'),
            'es': os.environ.get(
                'W2V_VECTOR_FILE_ES', '/datasets/wiki.es.vec'),
            'pt': os.environ.get(
                'W2V_VECTOR_FILE_PT', '/datasets/wiki.pt.vec'),
            'fr': os.environ.get(
                'W2V_VECTOR_FILE_FR', '/datasets/wiki.fr.vec'),
            'it': os.environ.get(
                'W2V_VECTOR_FILE_IT', '/datasets/wiki.it.vec'),
            'nl': os.environ.get(
                'W2V_VECTOR_FILE_NL', '/datasets/wiki.nl.vec')
        }
        self._server_port = os.environ.get('W2V_SERVER_PORT', '9090')

    @staticmethod
    def get_instance():
        if SvcConfig.__instance is None:
            SvcConfig.__instance = SvcConfig()
        return SvcConfig.__instance

    @property
    def vectors_file(self):
        return self._vectors_file

    @property
    def server_port(self):
        return int(self._server_port)
