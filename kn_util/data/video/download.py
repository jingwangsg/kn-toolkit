import re
import io
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
import yt_dlp
import requests
from contextlib import nullcontext
import tempfile
from yt_dlp.utils import parse_duration
import os.path as osp

from ...utils.error import SuppressStdoutStderr
from ...utils.system import buffer_keep_open, run_cmd


class FakeLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


class StorageLogger:
    def __init__(self):
        self.storage = defaultdict(list)

    def debug(self, msg):
        self.storage["debug"].append(msg)

    def warning(self, msg):
        self.storage["warning"].append(msg)

    def error(self, msg):
        self.storage["error"].append(msg)


def sub_to_dict(sub, dedupe=True, single=False) -> list:
    """Convert WebVTT to JSON, optionally removing duplicate lines"""
    import webvtt

    captions = webvtt.read_buffer(io.StringIO(sub))
    dicts = [{"start": c.start, "end": c.end, "lines": c.lines} for c in captions]
    if dedupe:
        dicts = []
        prev_line = None
        for c in captions:
            if any("<c>" in l for l in c.lines):
                continue
            # Collect lines that are not dupes
            not_dupe_lines = []
            for line in c.lines:
                if not line.strip():
                    continue
                if line != prev_line:
                    not_dupe_lines.append(line)
                prev_line = line
            if not_dupe_lines:
                dicts.append({"start": c.start, "end": c.end, "lines": not_dupe_lines})
    if single:
        for d in dicts:
            d["line"] = "\n".join(d.pop("lines"))
    return dicts


def _maybe_youtube_id(url):
    if url.startswith("http"):
        return re.match(r".*v=([^&]+)", url).group(1)
    else:
        return url


def download_youtube(
    youtube_id,
    video_path,
    video_format="worst[ext=mp4][height>=224]",
    quiet=True,
    logger=None,
    timestamp=None,
):
    # scale should be conditon like "<=224" or ">=224"
    youtube_id = _maybe_youtube_id(youtube_id)
    ydl_opts = {
        "ignoreerrors": True,
        "format": video_format,
        "outtmpl": video_path,
        "quiet": quiet,
        "noprogress": quiet,
        "logger": logger,
    }

    if timestamp is not None:
        st, ed = timestamp.split("-")

        parse_timestamp = lambda x: float("inf") if x in ("inf", "infinite") else parse_duration(x)
        ydl_opts["download_ranges"] = lambda _, __: [{"start_time": parse_timestamp(st), "end_time": parse_timestamp(ed)}]

    maybe_quiet = nullcontext()
    if quiet and logger is None:
        # completely suppress yt-dlp output
        ydl_opts["logger"] = FakeLogger()
        maybe_quiet = SuppressStdoutStderr()

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    with maybe_quiet, yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(url)

    return error_code


def download_youtube_as_bytes(youtube_id, video_format="worst[ext=mp4][height>=224]", quiet=True, logger=StorageLogger(), timestamp=None):
    youtube_id = _maybe_youtube_id(youtube_id)

    video_format_str = video_format.replace("*", "-").replace(":", "-").replace("/", "-")
    temp_path = osp.join(tempfile.gettempdir(), f"{youtube_id}.{video_format_str}.mp4")
    error_code = download_youtube(
        youtube_id=youtube_id,
        video_path=temp_path,
        video_format=video_format,
        quiet=quiet,
        logger=logger,
        timestamp=timestamp,
    )
    m = open(temp_path, "rb").read()
    run_cmd(f"rm -rf {temp_path}")

    return m, error_code


def download_yt_meta(
    url,
    yt_metadata_args: dict = {
        "writesubtitles": "first",
        "subtitleslangs": ["en"],
        "writeautomaticsub": True,
        "get_info": True,
    },
):
    """Return yt meta dict with meta data and/or subtitles
    yt_metadata_args is a dict of follwing format:
    yt_metadata_args = {
        'writesubtitles': 'first',
        'subtitleslangs': ['en'],
        'writeautomaticsub': True,
        'get_info': True
    }

    writesubtitles:    Whether to write subtitles for each provided language or just the first present
    writeautomaticsub: Write the automatically generated subtitles to a file
    subtitleslangs:    List of languages of the subtitles to download.
    get_info:          Whether to add info (title, description, tags etc) to the output.
    """

    write_subs = yt_metadata_args.get("writesubtitles", None)

    yt_metadata_args["skip_download"] = True
    yt_metadata_args["ignoreerrors"] = True
    yt_metadata_args["quiet"] = True

    info_dict, full_sub_dict = None, None

    with yt_dlp.YoutubeDL(yt_metadata_args) as yt:
        info_dict = yt.extract_info(url, download=False)

    if write_subs:
        full_sub_dict = {}
        for lang in yt_metadata_args["subtitleslangs"]:
            # if info_dict["requested_subtitles"] is None:
            #     import ipdb; ipdb.set_trace()
            if info_dict["requested_subtitles"] is None:
                break
            if lang not in info_dict["requested_subtitles"]:
                continue
            sub_url = info_dict["requested_subtitles"][lang]["url"]
            res = requests.get(sub_url, timeout=10)
            sub = io.TextIOWrapper(io.BytesIO(res.content)).read()
            full_sub_dict[lang] = sub_to_dict(sub)

            if write_subs == "first":
                break

    if yt_metadata_args["get_info"]:
        info_dict.pop("subtitles")
        info_dict.pop("requested_formats")
        info_dict.pop("formats")
        info_dict.pop("thumbnails")
        info_dict.pop("automatic_captions")
    else:
        info_dict = None

    yt_meta_dict = {"info": info_dict, "subtitles": full_sub_dict}

    return yt_meta_dict
