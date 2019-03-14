# -*- coding: utf-8 -*-

import os
import json
import yaml
import logging
import logging.config
import time
import traceback
import pathlib

import aiohttp
from aiohttp import web
import numpy

from word2vec.w2v import Word2Vec
from word2vec.svc_config import SvcConfig


def _get_logger():
    logger = logging.getLogger('word2vec.server')
    return logger


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


class Word2VecServer:
    def __init__(self):
        self.__w2v = None
        self.__mean = None
        self.__dim = None
        self.__loading = True
        self.logger = _get_logger()

    def load(self, path):
        wv = Word2Vec(path=path)
        self.logger.info("Loading vectors...")
        time1 = time.time()
        self.__w2v = wv.load_embeddings()
        self.__loading = False
        self.__dim = len(list(self.__w2v.values())[-1])
        self.__mean = wv.get_mean_norm(self.__w2v)
        time2 = time.time()
        self.logger.info(
            "Done loading vectors - took {}".format(time2 - time1))

    def gen_random_mean_norm_vector(self):
        tmp = numpy.random.normal(size=self.__dim).astype(numpy.float64)
        tmp /= numpy.linalg.norm(tmp) / self.__mean
        return tmp

    async def handle_reload(self, request):
        data = await request.json()
        if 'path' not in data:
            raise web.HTTPBadRequest()
        path = data['path']
        self.load(path)
        return web.Response()

    async def handle_request_multiple_words(self, request):
        """
        This endpoint handles a request that takes a JSON array of words, and returns
        a dictionary containing the vectorization of those words.
        Example:
        Request: {"words" : ["word1", "word2"]}
        Assuming we have the vectorisation for word1 but not for word2
        Response: {"vectors":{"word1":[...], "word2":null}}
        """

        data = await request.json()
        if 'words' not in data:
            raise web.HTTPBadRequest()
        words = data['words']
        self.logger.info("Request for {} words".format(len(words)))
        wordvec_dict = {}
        try:
            for word in words:
                vecs = self.__w2v.get(word)
                if vecs is not None:
                    wordvec_dict[word] = vecs
                else:
                    self.logger.info("unknown word {}".format(word))
            json_response = json.dumps({'vectors': wordvec_dict},
                                       cls=JsonEncoder)
            return web.json_response(body=json_response)
        except Exception:
            self.logger.exception("Error obtaining the vectors")
            raise

    async def handle_request_health(self, request):
        return web.Response(status=200)

    async def handle_request_unknown_words(self, request):
        data = await request.json()
        if 'words' not in data:
            raise web.HTTPBadRequest()
        words = data['words']
        self.logger.info("checking for unknown words from {} words".format(
            len(words)))
        try:
            unk_words = [w for w in words if w not in self.__w2v.keys()]
            json_response = json.dumps({'unk_words': unk_words},
                                       cls=JsonEncoder)
            return web.json_response(body=json_response)
        except Exception:
            self.logger.exception("Error obtaining unknown words")
            raise


LOGGING_CONFIG_TEXT = """
version: 1
root:
  level: DEBUG
  handlers: ['console']
formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "(asctime) (levelname) (name) (message)"
filters:
    w2vlogfilter:
        (): word2vec.server.W2vLogFilter
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    stream: ext://sys.stdout
    formatter: json
    filters: [w2vlogfilter]
"""


@web.middleware
async def log_error_middleware(request, handler):
    try:
        response = await handler(request)
    except aiohttp.web_exceptions.HTTPException:
        # assume if we're throwing this that it's already logged
        raise
    except Exception:
        _get_logger().exception("Unexpected exception in call")

        error_string = "Internal Server Error\n" + traceback.format_exc()
        raise aiohttp.web_exceptions.HTTPInternalServerError(text=error_string)
    return response


def initialize_web_app(app, w2v_server):
    app.middlewares.append(log_error_middleware)
    app.router.add_post('/words', w2v_server.handle_request_multiple_words)
    app.router.add_get('/health', w2v_server.handle_request_health)
    app.router.add_post('/unk_words', w2v_server.handle_request_unknown_words)
    app.router.add_post('/reload', w2v_server.handle_reload)


class W2vLogFilter(logging.Filter):
    def __init__(self):
        self.language = os.environ.get("W2V_LANGUAGE", "en")
        self.version = os.environ.get("W2V_VERSION", None)

    def filter(self, record):
        """Add language, and if available, the version"""
        record.w2v_language = self.language
        if self.version:
            record.w2v_version = self.version
        return True


def main():
    """Main function"""
    logging_config_file = os.environ.get("LOGGING_CONFIG_FILE", None)
    if logging_config_file:
        logging_config_path = pathlib.Path(logging_config_file)
        with logging_config_path.open() as file_handle:
            logging_config = yaml.safe_load(file_handle)
    else:
        logging_config = yaml.safe_load(LOGGING_CONFIG_TEXT)
    print("*** LOGGING CONFIG ***")
    print(logging_config)
    print("*** LOGGING CONFIG ***")
    logging.config.dictConfig(logging_config)

    config = SvcConfig.get_instance()
    server = Word2VecServer()
    server.load(config.vectors_file)

    app = web.Application()
    initialize_web_app(app, server)
    web.run_app(app, port=config.server_port)


if __name__ == '__main__':
    main()
