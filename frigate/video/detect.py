"""Manages camera object detection processes."""

import logging
import queue
import time
from datetime import UTC, datetime
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

import cv2

from frigate.camera import CameraMetrics, PTZMetrics
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import CameraConfig, DetectConfig, LoggerConfig, ModelConfig
from frigate.config.camera.camera import CameraTypeEnum
from frigate.config.camera.updater import (
    CameraConfigUpdateEnum,
    CameraConfigUpdateSubscriber,
)
from frigate.const import (
    PROCESS_PRIORITY_HIGH,
    REQUEST_REGION_GRID,
)
from frigate.motion import MotionDetector
from frigate.motion.improved_motion import ImprovedMotionDetector
from frigate.object_detection.base import RemoteObjectDetector
from frigate.ptz.autotrack import ptz_moving_at_frame_time
from frigate.track import ObjectTracker
from frigate.track.norfair_tracker import NorfairTracker
from frigate.track.tracked_object import TrackedObjectAttribute
from frigate.util.builtin import EventsPerSecond
from frigate.util.image import (
    FrameManager,
    SharedMemoryFrameManager,
    draw_box_with_label,
)
from frigate.util.object import (
    create_tensor_input,
    get_cluster_candidates,
    get_cluster_region,
    get_cluster_region_from_grid,
    get_min_region_size,
    get_startup_regions,
    inside_any,
    intersects_any,
    is_object_filtered,
    reduce_detections,
)
from frigate.util.process import FrigateProcess
from frigate.util.time import get_tomorrow_at_time

logger = logging.getLogger(__name__)


# FrigateProtect patch: extra regions the connector asks for (see /config/fp).
import json as _fp_json
import os as _fp_os

_FP_DIR = "/config/fp"
_fp_cache: dict[str, tuple[float, float, list]] = {}


def _fp_regions(cam, frame_time, frame_shape, regions, min_size):
    """Square regions (pixels) around the connector's boxes that are still wanted."""
    try:
        now = time.monotonic()
        c = _fp_cache.get(cam)
        if c is None or now - c[0] > 0.5:
            path = f"{_FP_DIR}/regions-{cam}.json"
            try:
                m = _fp_os.stat(path).st_mtime
            except OSError:
                m = 0.0
            if m == 0.0:
                items = []
            elif c is not None and m == c[1]:
                items = c[2]
            else:
                with open(path) as f:
                    items = _fp_json.load(f).get("regions", [])[:3]
            c = (now, m, items)
            _fp_cache[cam] = c
        if not c[2]:
            return []
        h, w = frame_shape[0], frame_shape[1]
        out = []
        for it in c[2]:
            if float(it.get("until", 0)) < frame_time:
                continue
            x1, y1, x2, y2 = (float(v) for v in it["box"])
            bx1, by1, bx2, by2 = x1 * w, y1 * h, x2 * w, y2 * h
            size = int(max(min_size, 1.5 * max(bx2 - bx1, by2 - by1)))
            size = min(size, w, h) // 4 * 4
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            rx = int(min(max(cx - size / 2, 0), w - size))
            ry = int(min(max(cy - size / 2, 0), h - size))
            r = (rx, ry, rx + size, ry + size)
            # skip it when Frigate already looks there at about the same detail
            if any(
                q[0] <= r[0] and q[1] <= r[1] and q[2] >= r[2] and q[3] >= r[3]
                and (q[2] - q[0]) <= 1.5 * size
                for q in regions
            ):
                continue
            out.append(r)
        return out
    except Exception:
        return []


# FrigateProtect patch: fisheye lenses. The detector learned "person" from upright photos, but a
# ceiling fisheye shows people lying sideways or upside down around the circle (feet toward the
# middle). Each region is turned so "away from the middle" points up before the detector sees
# it. Benchmark on the shop fisheye (docs/FISHEYE.md): people found 38 % -> 74 %, about the
# same detector work. On automatically for every camera whose picture is a round fisheye
# circle; /config/fp/fisheye-<camera>.json can override: {"mode": "off"} (e.g. a wall-mounted
# fisheye) or {"mode": "on", "circle": [cx, cy, r]} (normalised to the frame width/height).
import math as _fp_math

import numpy as _fp_np

from frigate.detectors.detector_config import PixelFormatEnum as _FpPixFmt
from frigate.util.image import yuv_region_2_bgr as _fp_bgr
from frigate.util.image import yuv_region_2_rgb as _fp_rgb

_fp_fish: dict[str, Any] = {}


def _fp_find_circle(frame, frame_shape):
    """(cx, cy, r) in pixels when the picture is a round fisheye image on black, else None."""
    h, w = frame_shape[0], frame_shape[1]
    y = cv2.resize(frame[0:h, 0:w], (256, max(16, int(256 * h / w))), interpolation=cv2.INTER_AREA)
    sh, sw = y.shape
    lit = (cv2.GaussianBlur(y, (5, 5), 0) > 20).astype(_fp_np.uint8)
    # drop thin bright things on the black border (the camera's clock and name)
    lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cs, _ = cv2.findContours(lit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    a = cv2.contourArea(c)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    # round (shop fisheye 0.84-0.91, normal cameras <= 0.61), big, and black outside it
    if a < 0.78 * _fp_math.pi * r * r or r < 0.4 * min(sw, sh):
        return None
    yy, xx = _fp_np.mgrid[0:sh, 0:sw]
    outside = (xx - cx) ** 2 + (yy - cy) ** 2 > (r * 1.04) ** 2
    if outside.mean() < 0.12 or _fp_np.median(y[outside]) > 20:  # lens border is pure black (16)
        return None
    # fit the circle to the blob's edge, ignoring edge points pulled out by text remnants
    pts = c.reshape(-1, 2).astype(_fp_np.float64)
    for _ in range(3):
        A = _fp_np.c_[2 * pts, _fp_np.ones(len(pts))]
        sol = _fp_np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)[0]
        cx, cy = sol[0], sol[1]
        r = _fp_math.sqrt(max(1.0, sol[2] + cx * cx + cy * cy))
        d = _fp_np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        keep = _fp_np.abs(d - _fp_np.median(d)) < 0.04 * _fp_np.median(d)
        if keep.sum() < 20:
            break
        pts = pts[keep]
    k = w / sw
    return (cx * k, cy * (h / sh), r * k)


def _fp_fisheye(cam, frame, frame_shape, now):
    """Circle for this camera (pixels) or None. Looks at the picture once a minute until it has
    seen a fisheye circle 3 times in a row (after that it keeps it); the override file is
    re-read every 10 s."""
    st = _fp_fish.setdefault(cam, {"circle": None, "seen": [], "next": 0.0, "file_t": 0.0, "file": None})
    if now - st["file_t"] > 10:
        st["file_t"] = now
        try:
            with open(f"{_FP_DIR}/fisheye-{cam}.json") as f:
                st["file"] = _fp_json.load(f)
        except (OSError, ValueError):
            st["file"] = None
    o = st["file"] or {}
    mode = o.get("mode", "auto")
    if mode == "off":
        return None
    h, w = frame_shape[0], frame_shape[1]
    if mode == "on" and o.get("circle"):
        cx, cy, r = (float(v) for v in o["circle"])
        return (cx * w, cy * h, r * w)
    if st["circle"] is None and now >= st["next"]:
        st["next"] = now + 60
        try:
            found = _fp_find_circle(frame, frame_shape)
        except Exception:
            found = None
        st["seen"] = st["seen"] + [found] if found is not None else []
        if len(st["seen"]) >= 3:
            st["circle"] = tuple(float(_fp_np.median([c[i] for c in st["seen"]])) for i in range(3))
        if st["circle"] is not None:
            logger.info(f"{cam}: fisheye lens found {tuple(int(v) for v in st['circle'])}, people are turned upright for the detector")
    if st["circle"] is None and mode == "on":
        return (w / 2, h / 2, min(w, h) / 2)
    return st["circle"]


def _fp_detect(cam, detect_config, object_detector, frame, model_config, region, objects_to_track, object_filters):
    """Frigate's detect(), but on a fisheye each region is turned upright first."""
    try:
        circ = _fp_fisheye(cam, frame, (detect_config.height, detect_config.width), time.monotonic())
    except Exception:
        circ = None
    if circ is None or model_config.input_pixel_format not in (_FpPixFmt.rgb, _FpPixFmt.bgr):
        return detect(detect_config, object_detector, frame, model_config, region, objects_to_track, object_filters)
    try:
        return _fp_detect_turned(circ, detect_config, object_detector, frame, model_config, region, objects_to_track, object_filters)
    except Exception as e:
        if not _fp_fish.get("_warned"):
            _fp_fish["_warned"] = True
            logger.warning(f"{cam}: turning fisheye regions failed ({e}); using plain regions")
        return detect(detect_config, object_detector, frame, model_config, region, objects_to_track, object_filters)


def _fp_detect_turned(circ, detect_config, object_detector, frame, model_config, region, objects_to_track, object_filters):
    fcx, fcy, fr = circ
    size = region[2] - region[0]
    px, py = (region[0] + region[2]) / 2, (region[1] + region[3]) / 2
    if _fp_math.hypot(px - fcx, py - fcy) < 0.12 * fr:
        return detect(detect_config, object_detector, frame, model_config, region, objects_to_track, object_filters)
    # an upright square crop that holds the turned region whatever the angle
    big = int(_fp_math.ceil(size * 1.42 / 4)) * 4
    bx, by = int(round(px - big / 2)), int(round(py - big / 2))
    outer = (bx, by, bx + big, by + big)
    tmp = (_fp_rgb if model_config.input_pixel_format == _FpPixFmt.rgb else _fp_bgr)(frame, outer)
    W, H = model_config.width, model_config.height
    phi = _fp_math.atan2(py - fcy, px - fcx)
    ey = _fp_np.array([-_fp_math.cos(phi), -_fp_math.sin(phi)])  # down = toward the middle
    ex = _fp_np.array([ey[1], -ey[0]])
    M = _fp_np.zeros((2, 3))
    M[:, 0] = ex * size / W
    M[:, 1] = ey * size / H
    M[:, 2] = _fp_np.array([big / 2, big / 2]) - (W / 2) * (size / W) * ex - (H / 2) * (size / H) * ey
    turned = cv2.warpAffine(tmp, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    detections = []
    fw, fh = detect_config.width, detect_config.height
    for d in object_detector.detect(_fp_np.expand_dims(turned, axis=0)):
        b = d[2]  # ymin, xmin, ymax, xmax of the model input, 0..1
        pts = _fp_np.array([[b[1] * W, b[0] * H, 1], [b[3] * W, b[0] * H, 1], [b[1] * W, b[2] * H, 1], [b[3] * W, b[2] * H, 1]]) @ M.T
        x_min = int(max(0, pts[:, 0].min() + bx))
        y_min = int(max(0, pts[:, 1].min() + by))
        x_max = int(min(fw - 1, pts[:, 0].max() + bx))
        y_max = int(min(fh - 1, pts[:, 1].max() + by))
        if x_min >= fw - 1 or y_min >= fh - 1 or x_max <= x_min or y_max <= y_min:
            continue
        width, height = x_max - x_min, y_max - y_min
        area = width * height
        det = (d[0], d[1], (x_min, y_min, x_max, y_max), area, width / max(1, height), outer)
        if is_object_filtered(det, objects_to_track, object_filters):
            continue
        detections.append(det)
    return detections


class CameraTracker(FrigateProcess):
    def __init__(
        self,
        config: CameraConfig,
        model_config: ModelConfig,
        labelmap: dict[int, str],
        detection_queue: Queue,
        detected_objects_queue,
        camera_metrics: CameraMetrics,
        ptz_metrics: PTZMetrics,
        region_grid: list[list[dict[str, Any]]],
        stop_event: MpEvent,
        log_config: LoggerConfig | None = None,
    ) -> None:
        super().__init__(
            stop_event,
            PROCESS_PRIORITY_HIGH,
            name=f"frigate.process:{config.name}",
            daemon=True,
        )
        self.config = config
        self.model_config = model_config
        self.labelmap = labelmap
        self.detection_queue = detection_queue
        self.detected_objects_queue = detected_objects_queue
        self.camera_metrics = camera_metrics
        self.ptz_metrics = ptz_metrics
        self.region_grid = region_grid
        self.log_config = log_config

    def run(self) -> None:
        self.pre_run_setup(self.log_config)
        frame_queue = self.camera_metrics.frame_queue
        frame_shape = self.config.frame_shape

        motion_detector = ImprovedMotionDetector(
            frame_shape,
            self.config.motion,
            self.config.detect.fps,
            name=self.config.name,
            ptz_metrics=self.ptz_metrics,
        )
        object_detector = RemoteObjectDetector(
            self.config.name,
            self.labelmap,
            self.detection_queue,
            self.model_config,
            self.stop_event,
        )

        object_tracker = NorfairTracker(self.config, self.ptz_metrics)

        frame_manager = SharedMemoryFrameManager()

        # create communication for region grid updates
        requestor = InterProcessRequestor()

        process_frames(
            requestor,
            frame_queue,
            frame_shape,
            self.model_config,
            self.config,
            frame_manager,
            motion_detector,
            object_detector,
            object_tracker,
            self.detected_objects_queue,
            self.camera_metrics,
            self.stop_event,
            self.ptz_metrics,
            self.region_grid,
        )

        # empty the frame queue
        logger.info(f"{self.config.name}: emptying frame queue")
        while not frame_queue.empty():
            (frame_name, _) = frame_queue.get(False)
            frame_manager.delete(frame_name)

        logger.info(f"{self.config.name}: exiting subprocess")


def detect(
    detect_config: DetectConfig,
    object_detector,
    frame,
    model_config: ModelConfig,
    region,
    objects_to_track,
    object_filters,
):
    tensor_input = create_tensor_input(frame, model_config, region)

    detections = []
    region_detections = object_detector.detect(tensor_input)
    for d in region_detections:
        box = d[2]
        size = region[2] - region[0]
        x_min = int(max(0, (box[1] * size) + region[0]))
        y_min = int(max(0, (box[0] * size) + region[1]))
        x_max = int(min(detect_config.width - 1, (box[3] * size) + region[0]))
        y_max = int(min(detect_config.height - 1, (box[2] * size) + region[1]))

        # ignore objects that were detected outside the frame
        if (x_min >= detect_config.width - 1) or (y_min >= detect_config.height - 1):
            continue

        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        ratio = width / max(1, height)
        det = (d[0], d[1], (x_min, y_min, x_max, y_max), area, ratio, region)
        # apply object filters
        if is_object_filtered(det, objects_to_track, object_filters):
            continue
        detections.append(det)
    return detections


def process_frames(
    requestor: InterProcessRequestor,
    frame_queue: Queue,
    frame_shape: tuple[int, int],
    model_config: ModelConfig,
    camera_config: CameraConfig,
    frame_manager: FrameManager,
    motion_detector: MotionDetector,
    object_detector: RemoteObjectDetector,
    object_tracker: ObjectTracker,
    detected_objects_queue: Queue,
    camera_metrics: CameraMetrics,
    stop_event: MpEvent,
    ptz_metrics: PTZMetrics,
    region_grid: list[list[dict[str, Any]]],
    exit_on_empty: bool = False,
):
    next_region_update = get_tomorrow_at_time(2)
    config_subscriber = CameraConfigUpdateSubscriber(
        None,
        {camera_config.name: camera_config},
        [
            CameraConfigUpdateEnum.detect,
            CameraConfigUpdateEnum.enabled,
            CameraConfigUpdateEnum.motion,
            CameraConfigUpdateEnum.objects,
        ],
    )

    fps_tracker = EventsPerSecond()
    fps_tracker.start()

    startup_scan = True
    stationary_frame_counter = 0
    camera_enabled = True

    region_min_size = get_min_region_size(model_config)

    attributes_map = model_config.attributes_map
    all_attributes = model_config.all_attributes

    # remove license_plate from attributes if this camera is a dedicated LPR cam
    if camera_config.type == CameraTypeEnum.lpr:
        attributes_map = {
            label: [attr for attr in attributes if attr != "license_plate"]
            for label, attributes in model_config.attributes_map.items()
        }
        all_attributes = [
            attr for attr in model_config.all_attributes if attr != "license_plate"
        ]

    while not stop_event.is_set():
        updated_configs = config_subscriber.check_for_updates()

        if "enabled" in updated_configs:
            prev_enabled = camera_enabled
            camera_enabled = camera_config.enabled

        if "motion" in updated_configs:
            motion_detector.config = camera_config.motion
            motion_detector.update_mask()

        if (
            not camera_enabled
            and prev_enabled != camera_enabled
            and camera_metrics.frame_queue.empty()
        ):
            logger.debug(
                f"Camera {camera_config.name} disabled, clearing tracked objects"
            )
            prev_enabled = camera_enabled

            # Clear norfair's dictionaries
            object_tracker.tracked_objects.clear()
            object_tracker.disappeared.clear()
            object_tracker.stationary_box_history.clear()
            object_tracker.positions.clear()
            object_tracker.track_id_map.clear()

            # Clear internal norfair states
            for trackers_by_type in object_tracker.trackers.values():
                for tracker in trackers_by_type.values():
                    tracker.tracked_objects = []
            for tracker in object_tracker.default_tracker.values():
                tracker.tracked_objects = []

        if not camera_enabled:
            time.sleep(0.1)
            continue

        if datetime.now().astimezone(UTC) > next_region_update:
            region_grid = requestor.send_data(REQUEST_REGION_GRID, camera_config.name)
            next_region_update = get_tomorrow_at_time(2)

        try:
            if exit_on_empty:
                frame_name, frame_time = frame_queue.get(False)
            else:
                frame_name, frame_time = frame_queue.get(True, 1)
        except queue.Empty:
            if exit_on_empty:
                logger.info("Exiting track_objects...")
                break
            continue

        camera_metrics.detection_frame.value = frame_time
        ptz_metrics.frame_time.value = frame_time

        frame = frame_manager.get(frame_name, (frame_shape[0] * 3 // 2, frame_shape[1]))

        if frame is None:
            logger.debug(
                f"{camera_config.name}: frame {frame_time} is not in memory store."
            )
            continue

        # look for motion if enabled
        motion_boxes = motion_detector.detect(frame)

        regions = []
        consolidated_detections = []

        # if detection is disabled
        if not camera_config.detect.enabled:
            object_tracker.match_and_update(frame_name, frame_time, [])
        else:
            # get stationary object ids
            # check every Nth frame for stationary objects
            # disappeared objects are not stationary
            # also check for overlapping motion boxes
            if stationary_frame_counter == camera_config.detect.stationary.interval:
                stationary_frame_counter = 0
                stationary_object_ids = []
            else:
                stationary_frame_counter += 1
                stationary_object_ids = [
                    obj["id"]
                    for obj in object_tracker.tracked_objects.values()
                    # if it has exceeded the stationary threshold
                    if obj["motionless_count"]
                    >= camera_config.detect.stationary.threshold
                    # and it hasn't disappeared
                    and object_tracker.disappeared[obj["id"]] == 0
                    # and it doesn't overlap with any current motion boxes when not calibrating
                    and not intersects_any(
                        obj["box"],
                        [] if motion_detector.is_calibrating() else motion_boxes,
                    )
                ]

            # get tracked object boxes that aren't stationary
            tracked_object_boxes = [
                (
                    # use existing object box for stationary objects
                    obj["estimate"]
                    if obj["motionless_count"]
                    < camera_config.detect.stationary.threshold
                    else obj["box"]
                )
                for obj in object_tracker.tracked_objects.values()
                if obj["id"] not in stationary_object_ids
            ]
            object_boxes = tracked_object_boxes + object_tracker.untracked_object_boxes

            # get consolidated regions for tracked objects
            regions = [
                get_cluster_region(
                    frame_shape, region_min_size, candidate, object_boxes
                )
                for candidate in get_cluster_candidates(
                    frame_shape, region_min_size, object_boxes
                )
            ]

            # only add in the motion boxes when not calibrating and a ptz is not moving via autotracking
            # the ptz timestamps are only maintained while autotracking is on, so gate
            # on the metric rather than trusting them to be reset otherwise
            ptz_moving = ptz_metrics.autotracker_enabled.value and (
                ptz_moving_at_frame_time(
                    frame_time,
                    ptz_metrics.start_time.value,
                    ptz_metrics.stop_time.value,
                )
            )

            if not motion_detector.is_calibrating() and not ptz_moving:
                # find motion boxes that are not inside tracked object regions
                standalone_motion_boxes = [
                    b for b in motion_boxes if not inside_any(b, regions)
                ]

                if standalone_motion_boxes:
                    motion_clusters = get_cluster_candidates(
                        frame_shape,
                        region_min_size,
                        standalone_motion_boxes,
                    )
                    motion_regions = [
                        get_cluster_region_from_grid(
                            frame_shape,
                            region_min_size,
                            candidate,
                            standalone_motion_boxes,
                            region_grid,
                        )
                        for candidate in motion_clusters
                    ]
                    regions += motion_regions

            # if starting up, get the next startup scan region
            if startup_scan:
                for region in get_startup_regions(
                    frame_shape, region_min_size, region_grid
                ):
                    regions.append(region)
                startup_scan = False

            # FrigateProtect patch: look where the connector saw something keep moving
            regions += _fp_regions(
                camera_config.name, frame_time, frame_shape, regions, region_min_size
            )

            # resize regions and detect
            # seed with stationary objects
            detections = [
                (
                    obj["label"],
                    obj["score"],
                    obj["box"],
                    obj["area"],
                    obj["ratio"],
                    obj["region"],
                )
                for obj in object_tracker.tracked_objects.values()
                if obj["id"] in stationary_object_ids
            ]

            for region in regions:
                detections.extend(
                    _fp_detect(  # FrigateProtect patch: fisheye regions turned upright
                        camera_config.name,
                        camera_config.detect,
                        object_detector,
                        frame,
                        model_config,
                        region,
                        camera_config.objects.track,
                        camera_config.objects.filters,
                    )
                )

            consolidated_detections = reduce_detections(frame_shape, detections)

            # if detection was run on this frame, consolidate
            if len(regions) > 0:
                tracked_detections = [
                    d for d in consolidated_detections if d[0] not in all_attributes
                ]
                # now that we have refined our detections, we need to track objects
                object_tracker.match_and_update(
                    frame_name, frame_time, tracked_detections
                )
            # else, just update the frame times for the stationary objects
            else:
                object_tracker.update_frame_times(frame_name, frame_time)

        # build detections
        detections = {}
        for obj in object_tracker.tracked_objects.values():
            detections[obj["id"]] = {**obj, "attributes": []}

        # assign each detected attribute to the best matching object.
        # iterate consolidated_detections once so attributes that appear under
        # multiple parent labels in attributes_map (e.g. license_plate is in
        # both "car" and "motorcycle") are not appended more than once
        all_objects: list[dict[str, Any]] = object_tracker.tracked_objects.values()
        detected_attributes = [
            TrackedObjectAttribute(d)
            for d in consolidated_detections
            if d[0] in all_attributes
        ]
        for attribute in detected_attributes:
            filtered_objects = filter(
                lambda o: attribute.label in attributes_map.get(o["label"], []),
                all_objects,
            )
            selected_object_id = attribute.find_best_object(filtered_objects)

            if selected_object_id is not None:
                detections[selected_object_id]["attributes"].append(
                    attribute.get_tracking_data()
                )

        # debug object tracking
        if False:
            bgr_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_YUV2BGR_I420,
            )
            object_tracker.debug_draw(bgr_frame, frame_time)
            cv2.imwrite(
                f"debug/frames/track-{'{:.6f}'.format(frame_time)}.jpg", bgr_frame
            )
        # debug
        if False:
            bgr_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_YUV2BGR_I420,
            )

            for m_box in motion_boxes:
                cv2.rectangle(
                    bgr_frame,
                    (m_box[0], m_box[1]),
                    (m_box[2], m_box[3]),
                    (0, 0, 255),
                    2,
                )

            for b in tracked_object_boxes:
                cv2.rectangle(
                    bgr_frame,
                    (b[0], b[1]),
                    (b[2], b[3]),
                    (255, 0, 0),
                    2,
                )

            for obj in object_tracker.tracked_objects.values():
                if obj["frame_time"] == frame_time:
                    thickness = 2
                    color = model_config.colormap.get(obj["label"], (255, 255, 255))
                else:
                    thickness = 1
                    color = (255, 0, 0)

                # draw the bounding boxes on the frame
                box = obj["box"]

                draw_box_with_label(
                    bgr_frame,
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                    obj["label"],
                    obj["id"],
                    thickness=thickness,
                    color=color,
                )

            for region in regions:
                cv2.rectangle(
                    bgr_frame,
                    (region[0], region[1]),
                    (region[2], region[3]),
                    (0, 255, 0),
                    2,
                )

            cv2.imwrite(
                f"debug/frames/{camera_config.name}-{'{:.6f}'.format(frame_time)}.jpg",
                bgr_frame,
            )
        # add to the queue if not full
        if detected_objects_queue.full():
            frame_manager.close(frame_name)
            continue
        else:
            fps_tracker.update()
            camera_metrics.process_fps.value = fps_tracker.eps()
            detected_objects_queue.put(
                (
                    camera_config.name,
                    frame_name,
                    frame_time,
                    detections,
                    motion_boxes,
                    regions,
                )
            )
            camera_metrics.detection_fps.value = object_detector.fps.eps()
            frame_manager.close(frame_name)

    motion_detector.stop()
    requestor.stop()
    config_subscriber.stop()
