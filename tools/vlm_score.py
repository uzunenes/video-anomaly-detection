"""Vision-language model baseline for one-class video anomaly detection (AnomalyRuler-style, open models, one VLM).

    python tools/vlm_score.py --data cache/ped2 --model Qwen/Qwen3-VL-8B-Instruct --out runs/vlm/ped2_q3vl8b

1. Induction (once per scene): the VLM describes --k normal TRAINING frames and turns the descriptions into the list
   of what normally appears in the scene (closed world: only what the normal frames show). Saved as rules.txt.
2. Deduction, --mode describe (default, AnomalyRuler's perception + reasoning): the VLM describes the test frame, then
   (text only) judges whether the description contains anything outside the normal list; --mode direct: one forward
   pass on the frame with the normal list. The frame score is log P(Yes) - log P(No) of the next token.
Frames come from the cache (<data>/test/<video>.npy, <data>/train/*.npy) or, with --frames-root / --train-videos, from
the original images and videos (e.g. ShanghaiTech in colour at full resolution; the induction and the test frames must
then come from the same source). Every --stride-th frame is scored; the others hold the latest score (causal). Writes scores.npz (vlm/<video>,
gt/<video>), rules.txt, descriptions.jsonl (describe mode) and timing.json (ms per frame at batch size 1 and at
--batch). Nothing is taken from the test videos except the frame being scored.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DESCRIBE = "Describe the people, vehicles, objects and activities visible in this surveillance frame in two sentences."
INDUCE = ("These are descriptions of NORMAL frames from one fixed surveillance camera:\n{desc}\n\n"
          "List, as short bullet points, the GENERAL kinds of objects, vehicles and activities that appear in these "
          "normal frames, e.g. 'pedestrians walking', 'people standing and talking', 'trash bins'. Do not list "
          "clothing, colours, ages or individual people. Only list what the descriptions mention; at most 15 bullets.")
PERCEIVE = ("Describe the people and objects in this surveillance frame and what each person is doing, in at most two "
            "sentences.")
JUDGE = ("Normal frames of this fixed surveillance camera contain only the following:\n{rules}\n\n"
         "Description of the current frame: {desc}\n\n"
         "Does the current frame contain any activity, object or vehicle whose KIND is not in the normal list? "
         "Ignore clothing, colours and the number of people. Answer Yes or No.")
ASK = ("Normal frames of this fixed surveillance camera contain only the following:\n{rules}\n\n"
       "Does this frame contain any activity, object or vehicle whose KIND is not in the normal list? "
       "Ignore clothing, colours and the number of people. Answer Yes or No.")


def to_pil(f: np.ndarray, side: int) -> Image.Image:
    """Frame -> RGB image whose longest side is `side` (small frames are enlarged: small VLMs miss small objects)."""
    im = Image.fromarray(f if f.ndim == 3 else np.repeat(f[..., None], 3, axis=2))
    s = side / max(im.size)
    return im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC) if s != 1 else im


class Frames:
    """Test frames of one video: the cache (N, H, W[, 3]) or the original images <root>/<video>/*.jpg (first n)."""

    def __init__(self, data: Path, name: str, root: str, n: int):
        self.files = sorted((Path(root) / name).glob("*.jpg"))[:n] if root else None
        self.arr = None if root else np.load(data / "test" / f"{name}.npy", mmap_mode="r")
        self.n = len(self.files) if root else len(self.arr)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.asarray(Image.open(self.files[i]).convert("RGB")) if self.files is not None else np.asarray(self.arr[i])


def normal_frames(data: Path, k: int, video_glob: str) -> list:
    """k normal training frames, spread over the training videos (original videos when video_glob is given)."""
    out = []
    if video_glob:
        import cv2
        vids = sorted(glob.glob(video_glob))
        for i in range(k):
            cap = cv2.VideoCapture(vids[i % len(vids)])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, (i * 7919) % max(n, 1))
            ok, f = cap.read()
            if ok:
                out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return out
    train = sorted((data / "train").glob("*.npy"))
    for i in range(k):
        v = np.load(train[i % len(train)], mmap_mode="r")
        out.append(np.asarray(v[(i * 7919) % len(v)]))
    return out


def load_model(name: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    try:
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(name)
    proc.tokenizer.padding_side = "left"
    return model.eval(), proc


def inputs_for(proc, convs, device):
    return proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                    return_tensors="pt", padding=True).to(device)


@torch.no_grad()
def generate_many(model, proc, convs, n=200) -> list[str]:
    x = inputs_for(proc, convs, model.device)
    out = model.generate(**x, max_new_tokens=n, do_sample=False)
    return [t.strip() for t in proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)]


def generate(model, proc, conv, n=200) -> str:
    return generate_many(model, proc, [conv], n)[0]


def answer_ids(tok, words):
    return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})


@torch.no_grad()
def log_odds(model, proc, convs, yes, no) -> np.ndarray:
    lp = torch.log_softmax(model(**inputs_for(proc, convs, model.device)).logits[:, -1].float(), -1)
    return (torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1)).cpu().numpy()


def score_frames(model, proc, images, rules, yes, no, mode):
    """-> (log-odds per frame, descriptions or None)."""
    if mode == "direct":
        convs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                               {"type": "text", "text": ASK.format(rules=rules)}]}] for im in images]
        return log_odds(model, proc, convs, yes, no), None
    desc = generate_many(model, proc, [[{"role": "user", "content": [
        {"type": "image", "image": im}, {"type": "text", "text": PERCEIVE}]}] for im in images], 80)
    convs = [[{"role": "user", "content": [{"type": "text", "text": JUDGE.format(rules=rules, desc=d)}]}] for d in desc]
    return log_odds(model, proc, convs, yes, no), desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("describe", "direct"), default="describe")
    ap.add_argument("--k", type=int, default=16, help="normal training frames for the induction")
    ap.add_argument("--side", type=int, default=720, help="longest image side given to the model (px)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score only this many test videos (smoke test)")
    ap.add_argument("--frames-root", default="", help="original test images <root>/<video>/*.jpg instead of the cache")
    ap.add_argument("--train-videos", default="", help="glob of original normal training videos for the induction")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = {k: np.asarray(v) for k, v in json.loads((data / "gt.json").read_text()).items()}
    names = sorted(gt)[:args.limit] if args.limit else sorted(gt)
    model, proc = load_model(args.model)
    tok = proc.tokenizer
    yes, no = answer_ids(tok, ["Yes", " Yes", "yes", " yes"]), answer_ids(tok, ["No", " No", "no", " no"])

    rules_file = out / "rules.txt"
    if rules_file.exists():
        rules = rules_file.read_text()
    else:                                                   # induction from k normal training frames, spread out
        desc = []
        for f in normal_frames(data, args.k, args.train_videos):
            desc.append("- " + generate(model, proc, [{"role": "user", "content": [
                {"type": "image", "image": to_pil(f, args.side)}, {"type": "text", "text": DESCRIBE}]}], 120))
        rules = generate(model, proc, [{"role": "user", "content": [
            {"type": "text", "text": INDUCE.format(desc="\n".join(desc))}]}], 400)
        rules_file.write_text(rules)
        (out / "normal_descriptions.txt").write_text("\n".join(desc))

    # latency at batch 1 (live camera) and throughput at --batch
    v0 = Frames(data, names[0], args.frames_root, len(gt[names[0]]))
    ims = [to_pil(np.asarray(v0[i]), args.side) for i in range(min(len(v0), 4 * args.batch))]
    score_frames(model, proc, ims[:1], rules, yes, no, args.mode)          # warm-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    for im in ims[:8]:
        score_frames(model, proc, [im], rules, yes, no, args.mode)
    torch.cuda.synchronize()
    ms1 = (time.perf_counter() - t) / min(8, len(ims)) * 1000

    scores, n_scored, t_all = {}, 0, 0.0
    dfile = open(out / "descriptions.jsonl", "w") if args.mode == "describe" else None
    for name in names:
        v = Frames(data, name, args.frames_root, len(gt[name]))
        idx = list(range(0, len(v), args.stride))
        s = np.zeros(len(v), np.float32)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for b in range(0, len(idx), args.batch):
            ib = idx[b:b + args.batch]
            s[ib], desc = score_frames(model, proc, [to_pil(np.asarray(v[i]), args.side) for i in ib], rules, yes, no,
                                       args.mode)
            if dfile:
                for i, d, x in zip(ib, desc, s[ib]):
                    dfile.write(json.dumps({"video": name, "frame": i, "score": round(float(x), 3), "desc": d}) + "\n")
        torch.cuda.synchronize()
        t_all += time.perf_counter() - t
        n_scored += len(idx)
        for i in range(len(v)):                               # causal hold between scored frames
            if i % args.stride:
                s[i] = s[i - i % args.stride]
        scores[name] = s
        print(f"{name}: {len(idx)} frames scored", flush=True)
    if dfile:
        dfile.close()
    np.savez(out / "scores.npz", **{f"vlm/{k}": x for k, x in scores.items()}, **{f"gt/{k}": gt[k] for k in names})
    timing = {"model": args.model, "mode": args.mode, "k": args.k, "side": args.side, "stride": args.stride, "batch": args.batch,
              "ms_per_frame_batch1": round(ms1, 1), "ms_per_frame_batched": round(1000 * t_all / max(n_scored, 1), 1),
              "frames_scored": n_scored, "gpu": torch.cuda.get_device_name(),
              "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
