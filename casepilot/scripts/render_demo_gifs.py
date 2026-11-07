"""Encode actual browser screenshots as GIFs; run after record_customer_demo.cjs --live."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


def font(size, *, bold=False, mono=False):
    system = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = (
        [str(system / "consola.ttf"), "DejaVuSansMono.ttf"]
        if mono
        else [str(system / ("segoeuib.ttf" if bold else "segoeui.ttf")), "DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def paragraph(draw, text, xy, *, width, face, color, spacing=8):
    x, y = xy
    line = ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=face) > width:
            draw.text((x, y), line, font=face, fill=color)
            y += face.size + spacing
            line = word
        else:
            line = candidate
    draw.text((x, y), line, font=face, fill=color)
    return y + face.size + spacing


def render_workflow(capture, output, storyboard, report):
    scenes = json.loads(storyboard.read_text(encoding="utf-8"))
    if not scenes or [s["stage"] for s in scenes] != sorted(s["stage"] for s in scenes):
        raise ValueError("Storyboard must contain scenes in workflow order")
    frames, provenance = [], []
    root = capture.resolve()
    stages = ["Browse data", "Verify customer", "Read order", "Review proposal", "Execute + verify"]
    for number, scene in enumerate(scenes):
        source = (root / scene["source"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Storyboard source must stay inside the capture")
        crop = tuple(scene["crop"])
        with Image.open(source) as im:
            if not (0 <= crop[0] < crop[2] <= im.width and 0 <= crop[1] < crop[3] <= im.height):
                raise ValueError("Crop outside screenshot")
            screen = im.convert("RGB").crop(crop)
        canvas = Image.new("RGB", (1280, 840), "#f1f5f9")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 1280, 88), fill="#102b3e")
        draw.text(
            (28, 14),
            "CASEPILOT  /  A complete agent workflow",
            font=font(28, bold=True),
            fill="white",
        )
        draw.text(
            (29, 52),
            "Real English UI capture  |  Synthetic Retail data  |  "
            "Enlarged views + editorial captions",
            font=font(16),
            fill="#bdd7e4",
        )
        for i, label in enumerate(stages, 1):
            x = 25 + (i - 1) * 248
            active = i == scene["stage"]
            draw.rounded_rectangle(
                (x, 104, x + 236, 147), radius=9, fill="#0b7866" if active else "#dfe7ed"
            )
            draw.text(
                (x + 11, 114),
                f"{i}. {label}",
                font=font(18, bold=active),
                fill="white" if active else "#4c6474",
            )
        draw.rounded_rectangle((24, 165, 506, 770), radius=14, fill="white")
        draw.text(
            (46, 185),
            f"STEP {number + 1:02d} / {len(scenes):02d}",
            font=font(16, bold=True),
            fill="#0b7866",
        )
        y = paragraph(
            draw, scene["title"], (46, 224), width=435, face=font(27, bold=True), color="#102b3e"
        )
        y = paragraph(
            draw,
            scene["explanation"],
            (46, y + 17),
            width=430,
            face=font(22),
            color="#344f60",
            spacing=10,
        )
        draw.rounded_rectangle((43, max(455, y + 20), 486, 720), radius=9, fill="#edf5f3")
        y = max(475, y + 37)
        draw.text((59, y), "OBSERVED IN THIS RECORDING", font=font(14, bold=True), fill="#0b7866")
        for line in scene["observed"]:
            y = paragraph(
                draw,
                line,
                (59, y + 30),
                width=410,
                face=font(17, mono=True),
                color="#153c36",
                spacing=3,
            )
        draw.rounded_rectangle(
            (526, 165, 1256, 770), radius=14, fill="white", outline="#cbd8e2", width=2
        )
        draw.text((546, 181), scene["view"], font=font(17, bold=True), fill="#344f60")
        screen = ImageOps.contain(screen, (690, 540), Image.Resampling.LANCZOS)
        canvas.paste(screen, (546 + (690 - screen.width) // 2, 218 + (540 - screen.height) // 2))
        draw.text(
            (28, 792),
            "Model proposes  >  Code verifies  >  Tool executes  >  "
            "Result updates the next decision",
            font=font(20, bold=True),
            fill="#102b3e",
        )
        frames.append(canvas)
        provenance.append(
            {**scene, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    palette = frames[0].quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    target = output / "agent-workflow.gif"
    durations = [scene.get("duration_ms", 3000) for scene in scenes]
    frames[0].save(
        target,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=1,
        optimize=False,
    )
    frames[-1].convert("RGB").save(output / "agent-workflow-result.png")
    manifest = {
        "source_recording": {key: value for key, value in report.items() if key != "clips"},
        "editing": (
            "Chronological real screenshots, cropped and enlarged with English editorial captions. "
            "Waiting omitted; no fabricated tool results or UI states."
        ),
        "file": target.name,
        "size": [1280, 840],
        "duration_seconds": sum(durations) / 1000,
        "bytes": target.stat().st_size,
        "scenes": provenance,
    }
    with Image.open(target) as check:
        if not check.is_animated or check.n_frames != len(scenes):
            raise ValueError("Workflow GIF did not retain every scene")
        manifest["frames"] = check.n_frames
    (output / "workflow-recording.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{target}: {len(frames)} scenes, {sum(durations) / 1000}s, {target.stat().st_size:,} bytes"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Folder containing recording.json")
    parser.add_argument("--output", type=Path, default=Path("assets/demos"))
    parser.add_argument(
        "--storyboard", type=Path, help="Render one annotated workflow from scene JSON"
    )
    args = parser.parse_args()
    report = json.loads((args.capture / "recording.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("Only a successful, verified UI capture can be published")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.storyboard:
        render_workflow(args.capture, args.output, args.storyboard, report)
        return
    artifacts = []
    clips = {item["name"]: item for item in report["clips"]}
    for clip in report["clips"]:
        frames, durations = [], []
        sources = [clips[name] for name in clip.get("context_clips", [])] + [clip]
        rows = [(item["name"], row) for item in sources for row in item["frames"]]
        for source_clip, row in rows:
            with Image.open(args.capture / source_clip / row["file"]) as source:
                frame = source.convert("RGB")
                if report.get("type") != "Actual English chat UI recording":
                    frame.thumbnail((1280, 900), Image.Resampling.LANCZOS)
                if len(frames) >= 2:
                    changes = ImageChops.difference(frame, frames[-1]).convert("L").histogram()
                    if sum(changes[12:]) < 1000:
                        durations[-1] = min(2500, durations[-1] + 700)
                        continue  # Omit near-identical waiting frames, not execution steps.
                frames.append(frame)
                durations.append(700)
        if len(frames) < 2:
            raise ValueError(f"Insufficient frames: {clip['name']}")
        # Share a palette so unchanged UI regions encode as small deltas.
        palette = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        target = args.output / f"{clip['name']}.gif"
        durations[0], durations[-1] = max(1400, durations[0]), 4000
        frames[0].save(
            target,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
            disposal=1,
        )
        with Image.open(target) as check:
            if not check.is_animated or check.n_frames < 2:
                raise ValueError(f"Not animated: {target}")
            artifacts.append(
                {
                    "file": target.name,
                    "frames": check.n_frames,
                    "size": list(check.size),
                    "bytes": target.stat().st_size,
                    "duration_seconds": sum(durations) / 1000,
                    "source_clips": [item["name"] for item in sources],
                }
            )
        print(f"{target}: {target.stat().st_size:,} bytes")
    provenance = {key: value for key, value in report.items() if key != "clips"} | {
        "editing": (
            "Actual UI screenshots in order, with no captions, overlays or reframing. "
            "Chat captures retain their original size. Near-identical waiting frames are "
            "condensed with holds capped at 2.5s; changed frames play at 0.7s, final frame 4s. "
            "No fabricated UI or responses."
        ),
        "artifacts": artifacts,
    }
    (args.output / "recording.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
