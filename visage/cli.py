"""Command line for the lab.

Examples
--------
python -m visage detect samples/north-light.jpg -o results/north-light.jpg
python -m visage bench
python -m visage track clip.mp4 -o results/tracked.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from visage import __version__
from visage.pipeline import analyze, draw
from visage.tracker import SortTracker
from visage.geometry import Detection


def _read(path: str):
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"could not read image: {path}")
    return image


def cmd_detect(args: argparse.Namespace) -> None:
    image = _read(args.image)
    result = analyze(image, use_haar=not args.no_haar, use_yunet=not args.no_yunet)
    public = {
        "width": result["width"],
        "height": result["height"],
        "raw": result["raw"],
        "faces": result["faces"],
    }
    text = json.dumps(public, indent=2)
    if args.json:
        print(text)
    else:
        print(f"{len(result['faces'])} face(s)  ({len(result['raw'])} raw boxes before fusion)")
        for face in result["faces"]:
            q = face["quality"]["score"]
            print(
                f"  {face['source']:12}  score {face['score']:.3f}  "
                f"skin {face['skin_ratio']:.2f}  quality {q:.2f}  "
                f"box {face['x']},{face['y']} {face['w']}x{face['h']}"
            )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), draw(image, result))
        print(f"wrote {out}", file=sys.stderr)


def cmd_bench(_args: argparse.Namespace) -> None:
    from visage.bench import main as bench_main

    bench_main()


def _detect_frame(frame, use_haar: bool, use_yunet: bool) -> list[Detection]:
    result = analyze(frame, use_haar=use_haar, use_yunet=use_yunet)
    dets = []
    for face in result["faces"]:
        dets.append(
            Detection(
                x=face["x"],
                y=face["y"],
                w=face["w"],
                h=face["h"],
                score=face["score"],
                source=face["source"],
            )
        )
    return dets


def cmd_track(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    tracker = SortTracker()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = _detect_frame(frame, use_haar=not args.no_haar, use_yunet=not args.no_yunet)
        tracks = tracker.update(dets)
        for track in tracks:
            if track.hits < 2 and track.age > 2:
                continue
            x, y, w, h = track.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 170, 210), 2)
            cv2.putText(
                frame,
                f"id {track.id}",
                (x, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 170, 210),
                1,
                cv2.LINE_AA,
            )
            if len(track.history) >= 2:
                for a, b in zip(track.history, track.history[1:]):
                    cv2.line(frame, a, b, (80, 170, 210), 1, cv2.LINE_AA)
        if writer is not None:
            writer.write(frame)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote {args.output} ({n} frames)")
    else:
        print(f"tracked {n} frames (pass -o to write a video)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visage", description="Multi-cue face analysis laboratory")
    parser.add_argument("--version", action="version", version=f"visage {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    detect = sub.add_parser("detect", help="detect faces in one image")
    detect.add_argument("image")
    detect.add_argument("-o", "--output", help="annotated image path")
    detect.add_argument("--json", action="store_true")
    detect.add_argument("--no-haar", action="store_true")
    detect.add_argument("--no-yunet", action="store_true")
    detect.set_defaults(func=cmd_detect)

    bench = sub.add_parser("bench", help="run the experiment log")
    bench.set_defaults(func=cmd_bench)

    track = sub.add_parser("track", help="track faces through a video file")
    track.add_argument("video")
    track.add_argument("-o", "--output")
    track.add_argument("--no-haar", action="store_true")
    track.add_argument("--no-yunet", action="store_true")
    track.set_defaults(func=cmd_track)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
