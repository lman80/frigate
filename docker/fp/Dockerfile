# FrigateProtect image: the official Frigate release + the FrigateProtect patched files.
# Same as the FrigateProtect NVR, which mounts these files over the upstream image.
# (Building Frigate 0.18.0 from source broke in Nov 2026: kaltura/nginx-vod-module, a
# build dependency, was deleted from GitHub.)
ARG BASE
FROM ${BASE}
COPY frigate/video/detect.py /opt/frigate/frigate/video/detect.py
COPY frigate/track/object_processing.py /opt/frigate/frigate/track/object_processing.py
