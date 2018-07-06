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
from svc_config import SvcConfig


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


class Word2VecLoaded(object):
    __w2v = None
    __mean = None
    __dim = None
    __loading = True

    @staticmethod
    def load(path):
        wv = Word2Vec(path=path)
        logger = _get_logger()
        logger.info("Loading vectors...")
        time1 = time.time()
        Word2VecLoaded.__w2v = wv.load_w2v()
        Word2VecLoaded.__loading = False
        Word2VecLoaded.__dim = len(list(Word2VecLoaded.__w2v.values())[-1])
        Word2VecLoaded.__mean = wv.get_mean_norm(Word2VecLoaded.__w2v)
        time2 = time.time()
        logger.info("Done loading vectors - took {}".format(time2 - time1))

    @staticmethod
    def get_w2v():
        return Word2VecLoaded.__w2v

    @staticmethod
    def is_loading():
        return Word2VecLoaded.__loading

    @staticmethod
    def get_mean():
        return Word2VecLoaded.__mean

    @staticmethod
    def get_dim():
        return Word2VecLoaded.__dim



"""
This endpoint handles a request that takes a JSON array of words, and returns
a dictionary containing the vectorization of those words.
Example:
Request: {"words" : ["word1", "word2"]}
Assuming we have the vectorisation for word1 but not for word2
Response: {"vectors":{"word1":[...], "word2":null}}
"""


async def handle_request_multiple_words(request):

    def gen_random_mean_norm_vector():
        dim = Word2VecLoaded.get_dim()
        mean_norm = Word2VecLoaded.get_mean()
        tmp = numpy.random.normal(size=dim).astype(numpy.float64)
        tmp /= numpy.linalg.norm(tmp) / mean_norm
        return tmp

    data = await request.json()
    if 'words' not in data:
        raise web.HTTPBadRequest()
    words = data['words']
    _get_logger().info("Request for {} words".format(len(words)))
    wordvec_dict = {}
    try:
        for word in words:
            if word.replace(".", "", 1).isdigit():
                word = 'NUM'
            vecs = Word2VecLoaded.get_w2v().get(word)
            if vecs is not None:
                wordvec_dict[word] = vecs
            else:
                wordvec_dict[word] = gen_random_mean_norm_vector()
        json_response = json.dumps({'vectors': wordvec_dict}, cls=JsonEncoder)
        return web.json_response(json_response)
    except:
        _get_logger().exception("Error obtaining the vectors")
        pass


async def handle_request_health(request):
    return web.Response(status=200)


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
    config = SvcConfig.get_instance()
    Word2VecLoaded.load(config.vectors_file)
    app = web.Application(loop=loop)
    app.router.add_post('/words', handle_request_multiple_words)
    app.router.add_get('/health', handle_request_health)
    web.run_app(app, port=config.server_port)


if __name__ == '__main__':
    main()
