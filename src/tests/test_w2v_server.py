import os
import numbers
from pathlib import Path

import pytest
from aiohttp import web

import word2vec.server

TEST_PATH = Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope="module")
def w2v_server():
    """Word2VecServer takes ages to initialize, so define a single Word2VecServer
    which will be reused in every test"""
    server = word2vec.server.Word2VecServer()
    server.load(str(TEST_PATH / "data_test_embedding"))
    return server


@pytest.fixture()
def cli(loop, test_client, w2v_server):
    """Defines the CLI test HTTP client which will start a test HTTP server.
    Will reuse the module level ner_server pytest fixture which is the slow bit to initialize"""
    web_app = web.Application(loop=loop)
    word2vec.server.initialize_web_app(web_app, w2v_server)
    return loop.run_until_complete(test_client(web_app))


async def test_server_root_404(cli):
    resp = await cli.get('/')
    assert resp.status == 404


async def test_server_health_200(cli):
    resp = await cli.get('/health')
    assert resp.status == 200


def check_vector_numeric(vector):
    assert isinstance(vector, list)
    assert len(vector) > 1  # at least 2 numbers in list
    for item in vector:
        assert isinstance(item, numbers.Number)


async def test_found_test_word(cli, mocker, w2v_server):
    TEST_WORD = "Preproduction"
    mocker.spy(w2v_server, "gen_random_mean_norm_vector")

    resp = await cli.post('/words', json={"words": [TEST_WORD]})
    assert resp.status == 200

    # check a random vector was not generated
    assert w2v_server.gen_random_mean_norm_vector.call_count == 0

    json_data = await resp.json()
    vectors = json_data["vectors"]

    # check the test word was found and that we have a list of numbers
    vectors_word1 = vectors[TEST_WORD]
    check_vector_numeric(vectors_word1)


async def test_unknown_test_word(cli, mocker, w2v_server):
    TEST_WORD = "frobble"
    mocker.spy(w2v_server, "gen_random_mean_norm_vector")
    resp = await cli.post('/words', json={"words": [TEST_WORD]})
    assert resp.status == 200

    # check a random vector was generated
    assert w2v_server.gen_random_mean_norm_vector.call_count == 0

    # check response
    json_data = await resp.json()
    vectors = json_data["vectors"]

    # check the test word was found and that we have a list of numbers
    assert len(vectors) == 0


async def test_unknown_test_word_2(cli, mocker, w2v_server):
    TEST_WORD = "frobble"
    resp = await cli.post('/unk_words', json={"words": [TEST_WORD]})
    assert resp.status == 200

    # check response
    json_data = await resp.json()
    unk_words = json_data["unk_words"]
    assert len(unk_words) == 1


async def test_multiple_test_words(cli):
    TEST_WORDS = ["MetroCard", "RockBand", "what"]
    resp = await cli.post('/unk_words', json={"words": TEST_WORDS})
    assert resp.status == 200

    json_data = await resp.json()
    unk_words = json_data["unk_words"]
    assert len(unk_words) == 1

    resp = await cli.post('/words', json={"words": TEST_WORDS[:2]})
    assert resp.status == 200

    # check response
    json_data = await resp.json()
    vectors = json_data["vectors"]

    # check the test words were found and that they have a list of numbers
    for test_word in TEST_WORDS[:2]:
        vector = vectors[test_word]
        check_vector_numeric(vector)
