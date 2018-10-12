import argparse
import os
from pathlib import Path
import hu_build.build_docker

DOCKER_TEMPLATE = """
FROM busybox
LABEL maintainer "Paul Annetts <paul@hutoma.ai>"

{to_be_replaced}

CMD exec /bin/sh -c "trap : TERM INT; (while true; do sleep 1000; done) & wait"
"""

SCRIPT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))


def main(args):
    file_name = args.input_file
    out_path = SCRIPT_PATH / "out"

    docker_line = "COPY {} /data/word2vec.v2.data.pkl".format(file_name, file_name)
    docker_file_content = DOCKER_TEMPLATE.format(to_be_replaced=docker_line)
    dockerfile = out_path / "Dockerfile"
    dockerfile.write_text(docker_file_content, encoding="utf8")
    image_name = 'word2vec_data_{}'.format(args.word2vec_variant)
    docker_image = hu_build.build_docker.DockerImage(
        out_path,
        'backend/{}'.format(image_name),
        image_tag=args.docker_tag,
        registry='eu.gcr.io/hutoma-backend')
    hu_build.build_docker.build_single_image(
        image_name, docker_image, push=args.docker_push)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Build dockerised image for an image')
    PARSER.add_argument('input_file', help='Input name of word2vec data PKL file')
    PARSER.add_argument('--docker-tag', help='Docker tag', default='1.0')
    PARSER.add_argument(
        '--docker-push', help='Push docker images to GCR', action="store_true")
    BUILD_ARGS = PARSER.parse_args()
    main(BUILD_ARGS)
