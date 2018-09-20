import os


class SvcConfig(object):

    __instance = None

    def __init__(self):
        self._vectors_file_en = os.environ.get(
            'W2V_VECTOR_FILE_EN', '/datasets/GoogleNews-vectors-negative300.bin')
        self._vectors_file_es = os.environ.get(
            'W2V_VECTOR_FILE_ES', '/datasets/GoogleNews-vectors-negative300.bin')
        self._vectors_file_pt = os.environ.get(
            'W2V_VECTOR_FILE_PT', '/datasets/GoogleNews-vectors-negative300.bin')
        self._vectors_file_fr = os.environ.get(
            'W2V_VECTOR_FILE_FR', '/datasets/GoogleNews-vectors-negative300.bin')
        self._vectors_file_it = os.environ.get(
            'W2V_VECTOR_FILE_IT', '/datasets/GoogleNews-vectors-negative300.bin')
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
