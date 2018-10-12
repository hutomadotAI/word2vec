import argparse
import os
import tqdm
import time
from pathlib import Path
import google.auth.transport.requests as g_requests
from google.cloud import storage
from google.resumable_media.requests import ResumableUpload

SCRIPT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))


class DataError(Exception):
    pass


def confirm_prompt(prompt_text):
    reply = input(prompt_text + " (y/n)")
    if not reply.lower() == "y":
        raise DataError("User aborted operation")


def main(args):
    file_location = Path(args.file_location)
    file_name = file_location.name
    local_file = file_location
    client = storage.Client()
    blob_folder = "word2vec_service/v2"
    bucket_name = "hutoma-datasets"
    bucket = client.get_bucket(bucket_name)
    blob_path = "{}/{}".format(blob_folder, file_name)
    blob = bucket.blob(blob_path)

    print("Operation {}: blob is {}, local file is {}".format(
        args.operation, blob_path, local_file))

    if args.operation == "download":
        if not blob.exists():
            raise DataError("Blob {} doesn't exist".format(blob_path))
        if local_file.exists():
            confirm_prompt("File {} exists, overwrite?".format(local_file))
        blob.download_to_filename(str(local_file))
    elif args.operation == "upload":
        if not local_file.exists():
            raise DataError("File {} doesn't exist".format(blob_path))
        if blob.exists():
            confirm_prompt("Blob {} exists, overwrite?".format(local_file))

        url = ("https://www.googleapis.com/upload/storage/v1/b/{bucket}" +
               "/o?uploadType=resumable").format(bucket=bucket_name)
        bytes_in_1MB = 1024 * 1024
        chunk_size = bytes_in_1MB  # 1MB
        upload = ResumableUpload(url, chunk_size)
        metadata = {"name": blob_path}
        content_type = "application/octet-stream"
        transport = g_requests.AuthorizedSession(
            credentials=client._credentials)

        with local_file.open("rb") as file_stream:
            response = upload.initiate(transport, file_stream, metadata,
                                       content_type)
            if response.status_code != 200:
                raise DataError("Failed to initiate upload")

            consecutive_errors = 0
            bytes_uploaded = 0
            with tqdm.tqdm(
                    unit="B",
                    total=upload.total_bytes,
                    unit_scale=True,
                    unit_divisor=1024) as progress_bar:
                while not upload.finished:
                    response = upload.transmit_next_chunk(transport)
                    if response.status_code == 308 or response.status_code == 200:
                        consecutive_errors = 0
                        progress_bar.update(upload.bytes_uploaded -
                                            bytes_uploaded)
                        bytes_uploaded = upload.bytes_uploaded
                    else:
                        consecutive_errors += 1
                        sleep_time = consecutive_errors * 5
                        print("Failure code {}, waiting {}s".format(response.status_code))
                        time.sleep(sleep_time)
                        if consecutive_errors > 10:
                            raise DataError("Failed to complete upload")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Build dockerised image for an image')
    PARSER.add_argument('file_location', help='Input name of word2vec data')
    PARSER.add_argument(
        'operation',
        help='Upload or download?',
        choices=['upload', 'download'])
    BUILD_ARGS = PARSER.parse_args()
    main(BUILD_ARGS)
