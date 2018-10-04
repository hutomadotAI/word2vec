# -*- coding: utf-8 -*-

import os
import json
from aiohttp import web
import numpy
import yaml
import logging
import logging.config
import asyncio
import time

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
    def __init__(self, vectors_file):
        self.__w2v = None
        self.__mean = None
        self.__dim = None
        self.__loading = True
        self.logger = _get_logger()
        self.vectors_file = vectors_file

    def load(self, language='en'):
        try:
            path = self.vectors_file[language]
            self.logger.info("path: {}".format(path))
        except:
            NotImplementedError("language not yet implemented")
        self.logger.info("Loading vectors for {} from {}...".format(language, path))
        wv = Word2Vec(path=path)
        time1 = time.time()
        self.__w2v = wv.load_w2v()
        self.__loading = False
        self.__dim = len(list(self.__w2v.values())[-1])
        self.__mean = wv.get_mean_norm(self.__w2v)
        time2 = time.time()
        self.logger.info("Done loading vectors - took {}".format(time2 - time1))

    def gen_random_mean_norm_vector(self):
        tmp = numpy.random.normal(size=self.__dim).astype(numpy.float64)
        tmp /= numpy.linalg.norm(tmp) / self.__mean
        return tmp

    async def reload(self, request):
        """
        this endpoint reloads a language model
        """
        url = request.url
        language = url.query.get('language')
        self.load(language=language)
        return web.Response(status=200)

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
            json_response = json.dumps({'vectors': wordvec_dict}, cls=JsonEncoder)
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
        self.logger.info("checking for unknown words from {} words".format(len(words)))
        try:
            unk_words = [w for w in words if w not in self.__w2v.keys()]
            json_response = json.dumps({'unk_words': unk_words}, cls=JsonEncoder)
            return web.json_response(body=json_response)
        except Exception:
            self.logger.exception("Error obtaining unknown words")
            raise


LOGGING_CONFIG_TEXT = """
version: 1
root:
  level: DEBUG
  handlers: ['console' ,'elastic']
formatters:
  default:
    format: "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s"
    datefmt: "%Y%m%d_%H%M%S"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    stream: ext://sys.stdout
    formatter: default
  elastic:
    class: hu_logging.HuLogHandler
    level: INFO
    log_path: /tmp/hu_log
    log_tag: WORD2VEC
    es_log_index: ai-word2vec-v1
    multi_process: False
"""


def initialize_web_app(app, w2v_server):
    app.router.add_post('/words', w2v_server.handle_request_multiple_words)
    app.router.add_get('/health', w2v_server.handle_request_health)
    app.router.add_post('/unk_words', w2v_server.handle_request_unknown_words)
    app.router.add_get('/reload', w2v_server.reload)


def main():
    """Main function"""
    logging_config = yaml.load(LOGGING_CONFIG_TEXT)
    logging_config['handlers']['elastic']['elastic_search_url'] = \
        os.environ.get('LOGGING_ES_URL', None)
    log_tag = os.environ.get('LOGGING_ES_TAG', None)
    if log_tag:
        logging_config['handlers']['elastic']['log_tag'] = log_tag
    logging.config.dictConfig(logging_config)
    env_w2v_language = os.environ.get('W2V_LANGUAGE', 'en')

    loop = asyncio.get_event_loop()
    config = SvcConfig.get_instance()
    server = Word2VecServer(config.vectors_file)
    server.load(language=env_w2v_language)

    app = web.Application(loop=loop)
    initialize_web_app(app, server)
    web.run_app(app, port=config.server_port)


if __name__ == '__main__':
    main()
