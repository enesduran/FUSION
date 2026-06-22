"""Concatenate 2 or 4 video / GIF files into a single output via ffmpeg.

Layouts (auto-selected by input count):
    - 2 inputs -> stacked vertically (1 column, 2 rows).
    - 4 inputs -> 2x2 grid (2 columns, 2 rows; reading order = left-to-right,
      top-to-bottom).

All inputs must share the same extension (``.mp4`` for all, or ``.gif`` for
all). Inputs that don't already match the first input's resolution are
rescaled before stacking so the filters' size-equality requirements hold.

Examples
--------
    # Stack two clips vertically.
    python src/scripts/concat_media.py top.mp4 bottom.mp4 -o stacked.mp4

    # 2x2 grid of four GIFs.
    python src/scripts/concat_media.py a.gif b.gif c.gif d.gif -o grid.gif
"""

import argparse
import json
import os
import subprocess
import sys


def run(cmd, **kwargs):
    """Run a subprocess command and surface its stderr on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def probe_dimensions(path):
    """Return (width, height) for the first video stream in ``path``."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json', path,
    ]
    result = run(cmd)
    info = json.loads(result.stdout)
    streams = info.get('streams') or []
    if not streams:
        raise SystemExit(f"no video stream found in {path}")
    return int(streams[0]['width']), int(streams[0]['height'])


def build_filter(n_inputs, width, height):
    """Build an ffmpeg ``-filter_complex`` string for the requested layout."""
    scaled = []
    for i in range(n_inputs):
        # Scale + reset SAR so vstack/hstack accept the streams unconditionally.
        scaled.append(f"[{i}:v]scale={width}:{height},setsar=1[v{i}]")

    if n_inputs == 2:
        layout = "[v0][v1]vstack=inputs=2[out]"
    elif n_inputs == 4:
        layout = (
            "[v0][v1]hstack=inputs=2[top];"
            "[v2][v3]hstack=inputs=2[bot];"
            "[top][bot]vstack=inputs=2[out]"
        )
    else:  # pragma: no cover — argparse already gates this.
        raise SystemExit(f"unsupported input count: {n_inputs}")

    return ";".join(scaled + [layout])


def concat_mp4(inputs, out_path):
    width, height = probe_dimensions(inputs[0])
    filter_complex = build_filter(len(inputs), width, height)
    cmd = ['ffmpeg', '-y']
    for path in inputs:
        cmd += ['-i', path]
    cmd += [
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        '-movflags', '+faststart',
        out_path,
    ]
    run(cmd)


def concat_gif(inputs, out_path):
    width, height = probe_dimensions(inputs[0])
    layout_filter = build_filter(len(inputs), width, height)
    # Two-pass with a palette gives noticeably better gif quality than letting
    # ffmpeg pick a default 256-colour palette from a single frame.
    filter_complex = (
        f"{layout_filter};"
        "[out]split=2[a][b];"
        "[a]palettegen=stats_mode=diff[palette];"
        "[b][palette]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[final]"
    )
    cmd = ['ffmpeg', '-y']
    for path in inputs:
        cmd += ['-i', path]
    cmd += [
        '-filter_complex', filter_complex,
        '-map', '[final]',
        '-loop', '0',
        out_path,
    ]
    run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('inputs', nargs='+',
                        help='2 or 4 input files (all .mp4 or all .gif).')
    parser.add_argument('-o', '--output', required=True,
                        help='Output path. Extension must match the inputs.')
    args = parser.parse_args()

    n = len(args.inputs)
    if n not in (2, 4):
        raise SystemExit(f"need 2 or 4 inputs, got {n}")

    exts = {os.path.splitext(p)[1].lower() for p in args.inputs}
    if len(exts) != 1:
        raise SystemExit(f"all inputs must share an extension, got {sorted(exts)}")
    ext = next(iter(exts))
    if ext not in ('.mp4', '.gif'):
        raise SystemExit(f"unsupported extension {ext}; expected .mp4 or .gif")

    out_ext = os.path.splitext(args.output)[1].lower()
    if out_ext != ext:
        raise SystemExit(
            f"output extension {out_ext} does not match input extension {ext}"
        )

    for p in args.inputs:
        if not os.path.isfile(p):
            raise SystemExit(f"input file not found: {p}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    layout = 'vstack (2x1)' if n == 2 else '2x2 grid'
    print(f"[concat_media] {n} inputs -> {layout} -> {args.output}")

    if ext == '.mp4':
        concat_mp4(args.inputs, args.output)
    else:
        concat_gif(args.inputs, args.output)

    print(f"[concat_media] wrote {args.output}")


if __name__ == '__main__':
    main()
