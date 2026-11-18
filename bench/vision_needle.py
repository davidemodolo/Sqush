#!/usr/bin/env python3
# Vision needle test: an image with a random code and a colored circle at the start of the prompt,
# a long filler, then a question about the image. Needs the server started with --vision and Pillow.
#
#   python3 bench/vision_needle.py --reps 0      # short baseline
#   python3 bench/vision_needle.py --reps 560    # ~200k tokens, crosses both KV conversions
import argparse, base64, io, random, string, sys

from PIL import Image, ImageDraw, ImageFont

from common import DEFAULT_HOST, chat, filler, print_timings

COLORS = [((220, 30, 30), "red"), ((30, 160, 40), "green"), ((30, 60, 220), "blue")]


def make_image(code, rgb):
    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
    except OSError:
        font = ImageFont.load_default(size=64)
    d.text((40, 40), code, fill="black", font=font)
    d.ellipse((220, 180, 420, 380), fill=rgb)

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default="qwen3.8-27b-vision")
    ap.add_argument("--reps", type=int, default=0)
    args = ap.parse_args()

    code = "".join(random.choice(string.ascii_uppercase) for _ in range(4)) + "-" + str(random.randint(1000, 9999))
    rgb, color = random.choice(COLORS)

    content = [
        {"type": "text", "text": "Look carefully at this image, it will be asked about at the end."},
        {"type": "image_url", "image_url": {"url": make_image(code, rgb)}},
        {"type": "text", "text": filler(args.reps) + "\n\nQuestion about the image at the start: what code is "
                                 "written in it, and what color is the circle? Answer as: CODE, COLOR"},
    ]

    answer, r, wall = chat(args.host, args.model, [{"role": "user", "content": content}], max_tokens=32)
    print_timings(r, wall)
    print(f"expected: {code}, {color}")
    print(f"answer:   {answer[:120]!r}")

    ok_code, ok_color = code in answer.upper(), color in answer.lower()
    ok = ok_code and ok_color
    print("RESULT: " + ("PASS" if ok else f"FAIL (code {'ok' if ok_code else 'wrong'}, color {'ok' if ok_color else 'wrong'})"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
