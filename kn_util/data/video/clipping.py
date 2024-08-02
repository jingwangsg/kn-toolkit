import os, os.path as osp
from ffmpy import FFmpeg
import tempfile

from kn_util.utils.system import run_cmd
from kn_util.data.video.load import probe_meta


def cut_video_clips_ffselect(video_path, timesteps, output_dir, suffix_format="_{:02d}.mp4", with_audio=False, filter_kwargs=None):
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(timesteps[0][0], int) and isinstance(timesteps[0][1], int):
        template = "[0:v]trim=start_frame={start}:end_frame={end}"
        if filter_kwargs is not None:
            template += "," + filter_kwargs
        template += ",setpts=PTS-STARTPTS[v{i}]"
        if with_audio:
            template += ";[0:a]atrim=start_frame={start}:end_frame={end},asetpts=PTS-STARTPTS[a{i}]"
    else:
        template = "[0:v]trim=start={start}:end={end}"
        if filter_kwargs is not None:
            template += "," + filter_kwargs
        template += ",setpts=PTS-STARTPTS[v{i}]"
        if with_audio:
            template += ";[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]"
    filename = osp.basename(video_path)
    output_path_template = osp.join(output_dir, filename.replace(".mp4", suffix_format))
    filter_kwargs = [
        "-filter_complex",
        "'",
        ";".join([template.format(start=start, end=end, i=i) for (i, (start, end)) in enumerate(timesteps)]),
        "'",
    ]
    for i in range(len(timesteps)):
        filter_kwargs.extend(["-map", f"[v{i}]"])
        if with_audio:
            filter_kwargs.extend(["-map", f"[a{i}]"])
        filter_kwargs.append(output_path_template.format(i))

    global_options = "-hide_banner -loglevel error -y"
    ff = FFmpeg(
        inputs={video_path: None},
        outputs={None: filter_kwargs},
        global_options=global_options,
    )
    print(ff.cmd)
    run_cmd(ff.cmd)


def cut_video_clips_ffloop(video_path, timesteps, output_dir, suffix_format="_{:02d}.mp4", with_audio=False, filter_kwargs=None):
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(timesteps[0][0], int) and isinstance(timesteps[0][1], int):
        timesteps_sec = []
        video_meta = probe_meta(video_path)
        fps = video_meta["fps"]
        for start_frame, end_frame in timesteps:
            timesteps_sec.append((start_frame / fps, end_frame / fps))
        timesteps = timesteps_sec

    filter_kwargs = "-vf " + filter_kwargs if filter_kwargs is not None else ""
    for i, (start, end) in enumerate(timesteps):
        output_path = osp.join(output_dir, osp.basename(video_path).replace(".mp4", suffix_format.format(i)))
        ff = FFmpeg(
            inputs={video_path: None},
            outputs={output_path: f"-ss {start} -to {end}" + filter_kwargs},
            global_options="-hide_banner -loglevel error -y",
        )
        run_cmd(ff.cmd)


def cut_video_clips_ffsegment(video_path, timesteps, output_dir, suffix_format="_{:02d}.mp4", with_audio=False):
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(timesteps[0][0], int) and isinstance(timesteps[0][1], int):
        timesteps_sec = []
        video_meta = probe_meta(video_path)
        fps = video_meta["fps"]
        for start_frame, end_frame in timesteps:
            timesteps_sec.append((start_frame / fps, end_frame / fps))
        timesteps = timesteps_sec

    segment_times = []
    for start, end in timesteps:
        if len(segment_times) > 0 and segment_times[-1] == start:
            segment_times += [end]
        else:
            segment_times += [start, end]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = osp.join(tmpdir, "clip%d.mp4")
        ff = FFmpeg(
            inputs={video_path: None},
            outputs={output_path: ["-f", "segment", "-segment_times", ",".join(map(str, segment_times))]},
            global_options="-hide_banner -loglevel error -y",
        )
        print(ff.cmd)
        run_cmd(ff.cmd)

        timesteps_cnt = 0

        for i in range(len(segment_times) - 1):
            st, ed = segment_times[i], segment_times[i + 1]
            if timesteps[timesteps_cnt][0] == st and timesteps[timesteps_cnt][1] == ed:
                cur_output_path = output_path.replace("%d", str(i))
                target_path = osp.join(output_dir, osp.basename(video_path).replace(".mp4", suffix_format.format(timesteps_cnt)))
                run_cmd(f"mv {cur_output_path} {target_path}")
                timesteps_cnt += 1
                if timesteps_cnt == len(timesteps):
                    break


cut_video_clips = cut_video_clips_ffselect

if __name__ == "__main__":
    import time

    st = time.time()
    cut_video_clips_ffselect(
        "miradata/v6IeXGQ92aLE.mp4",
        [(732.0, 740.0), (120.0, 140.0), (160.0, 180.0)],
        output_dir="output1",
        suffix_format="_{:02d}.mp4",
        with_audio=False,
        filter_kwargs="scale=512:320",
    )
    print(time.time() - st)
