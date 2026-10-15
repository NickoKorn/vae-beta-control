"""Resume only the TCGA members of the official AUTOENCODIX Zenodo archive.

Uses bounded HTTP range requests, checks Content-Range, streams decompression
and validates archive CRC/size before publishing each parquet. Existing valid
files and partial downloads are retained. No training is started.
"""
import argparse
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
import zlib


def file_crc(path):
    crc = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            crc = zlib.crc32(block, crc)
    return crc


def fetch(source_path, destination):
    source = json.loads(Path(source_path).read_text())
    destination = Path(destination)
    cache = destination / '.download'
    cache.mkdir(parents=True, exist_ok=True)
    for member in source['members']:
        target = destination / member['name']
        if Path(member['name']).name != member['name']:
            raise ValueError('Archive member must be a filename')
        if target.exists() and target.stat().st_size == member['size'] and file_crc(target) == member['crc']:
            print('Verified:', target.name, flush=True)
            continue
        partial = cache / (member['name'] + '.zipmember')
        total = member['end_offset'] - member['offset'] + 1
        retries = 0
        while not partial.exists() or partial.stat().st_size < total:
            have = partial.stat().st_size if partial.exists() else 0
            start = member['offset'] + have
            end = min(member['end_offset'], start + 16 * 1024 * 1024 - 1)
            request = Request(source['url'], headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'AUTOENCODIX-TCGA-recovery/1'})
            try:
                with urlopen(request, timeout=60) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise ValueError('Server did not honour the requested byte range')
                    remaining = end - start + 1
                    with partial.open('ab') as output:
                        while remaining:
                            block = response.read(min(1024 * 1024, remaining))
                            if not block:
                                raise OSError('Incomplete range response')
                            output.write(block)
                            output.flush()
                            remaining -= len(block)
                retries = 0
                print(f'{target.name}: {partial.stat().st_size / total:.1%}', flush=True)
            except (OSError, TimeoutError) as error:
                retries += 1
                print(f'Retaining partial download after {error}; retry {retries}', flush=True)
                if retries > 10:
                    raise
                time.sleep(min(30, retries * 3))
        with partial.open('rb') as stream, target.with_suffix('.tmp').open('wb') as output:
            header = struct.unpack('<4s5H3I2H', stream.read(30))
            if header[0] != b'PK\x03\x04' or header[3] != 8:
                raise ValueError('Unexpected ZIP member header/compression')
            stream.read(header[-2] + header[-1])
            decoder = zlib.decompressobj(-15)
            remaining, crc, size = member['compressed_size'], 0, 0
            while remaining:
                block = stream.read(min(256 * 1024, remaining))
                if not block:
                    raise ValueError('Truncated ZIP member')
                remaining -= len(block)
                decoded = decoder.decompress(block)
                output.write(decoded);crc = zlib.crc32(decoded, crc);size += len(decoded)
            decoded = decoder.flush()
            output.write(decoded);crc = zlib.crc32(decoded, crc);size += len(decoded)
        if crc != member['crc'] or size != member['size'] or not decoder.eof:
            raise ValueError('Archive CRC/size verification failed; original target unchanged')
        target.with_suffix('.tmp').replace(target)
        print('Extracted and verified:', target, flush=True)
    (destination / 'source.json').write_text(json.dumps(source, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    fetch(args.source, args.destination)
