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

from word2vec import Word2Vec


def _get_logger():
    logger = logging.getLogger('svclass.server')
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


class Word2VecLoaded(object):
    __w2v = None
    __loading = True

    @staticmethod
    def load(path):
        wv = Word2Vec(path=path)
        logger = _get_logger()
        logger.info("Loading vectors...")
        time1 = time.time()
        Word2VecLoaded.__w2v = wv.load_w2v()
        Word2VecLoaded.__loading = False
        time2 = time.time()
        logger.info("Done loading vectors - took {}".format(time2-time1))

    @staticmethod
    def get_w2v():
        return Word2VecLoaded.__w2v

    @staticmethod
    def is_loading():
        return Word2VecLoaded.__loading


async def handle_web_request(request):
    if Word2VecLoaded.is_loading():
        raise web.HTTPServiceUnavailable
    word = request.match_info.get('word', "")
    word_vectors = Word2VecLoaded.get_w2v().get(word)
    if word_vectors is None:
        _get_logger().info("Requested vectors for word '{}' - not found".format(word))
        return web.json_response(json.dumps({'word': word, 'vectors': None}))
    else:
        _get_logger().info("Requested vectors for word '{}'".format(word))
        json_response = json.dumps({'word': word, 'vectors': word_vectors}, cls=JsonEncoder)
        return web.json_response(json_response)


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
    log_tag: WNET
    es_log_index: ai-svclass-v1
    multi_process: False
"""


def main():
    """Main function"""
    logging_config = yaml.load(LOGGING_CONFIG_TEXT)
    logging_config['handlers']['elastic']['elastic_search_url'] = \
        os.environ.get('LOGGING_ES_URL', None)
    log_tag = os.environ.get('LOGGING_ES_TAG', None)
    if log_tag:
        logging_config['handlers']['elastic']['log_tag'] = log_tag
    logging.config.dictConfig(logging_config)

    loop = asyncio.get_event_loop()
    vectors_file = os.environ.get("W2V_VECTOR_FILE", "/datasets/GoogleNews-vectors-negative300.bin")
    server_port = int(os.environ.get("W2V_SERVER_PORT", 9090))
    Word2VecLoaded.load(vectors_file)
    app = web.Application(loop=loop)
    app.router.add_get('/{word}', handle_web_request)
    web.run_app(app, port=server_port)


if __name__ == '__main__':
    main()
