import streamlit as st
import glob
from argparse import ArgumentParser
import os, os.path as osp

parser = ArgumentParser()
parser.add_argument("video_root", type=str)
args = parser.parse_args()

video_root = args.video_root
st.title(f"Video Gallery \n`{video_root}`")


@st.cache_data
def load_video_files():
    video_files = glob.glob(f"{video_root}/**/*.mp4") + glob.glob(f"{video_root}/**/*.avi")
    return video_files


video_files = load_video_files()
filename = st.text_input("Filename", "")

view_videos = []

nrow = st.slider("Number of videos per row", 1, 10, 3)
ncol = st.slider("Number of videos per column", 1, 10, 4)
nvideo_per_page = nrow * ncol

if filename == "":
    view_videos = video_files
else:
    for video_file in video_files:
        if filename in video_file:
            view_videos.append(video_file)

num_page = (len(view_videos) + nvideo_per_page - 1) // nvideo_per_page
npage = st.number_input(f"Page ({num_page})", 1, num_page + 1, 1)

groups = []
for i in range(nvideo_per_page * (npage - 1), nvideo_per_page * (npage - 1) + nvideo_per_page, nrow):
    groups.append(view_videos[i : i + nrow])

for group in groups:
    if len(group) == 0:
        continue
    cols = st.columns(len(group))
    for col, video_file in zip(cols, group):
        filename = osp.basename(video_file)
        col.video(video_file)
        col.text(filename)
