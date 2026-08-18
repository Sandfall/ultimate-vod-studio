#!/usr/bin/env python3
"""Train the 4-class m_char portrait classifier (SF / BR / GN / not_m_char) for the deployment cascade.

Reads an ImageFolder-style dataset: `{dataset}/{SF,BR,GN,
not_m_char}/*.png`, plus an optional `groups.csv` (filename -> group) so cross-validation can split
player-DISJOINT (an m_char player's crops never straddle train/val, so the score reflects unseen
players, not memorized faces). Negatives are grouped per-image. The crops are HUD portrait
tiles (the small fighter-face boxes in the in-game HUD); any ImageFolder tree in this shape,
with or without `groups.csv`, trains.

Recipe (preserved from the v2 model that fixed the SF<->GN color
shortcut): ResNet-18 (ImageNet init, only layer4+fc fine-tuned), 224px,
class-weighted CrossEntropy (not_m_char dwarfs each m_char class), and augmentation aimed at the real
deployment nuisances AND the spurious cues v1 latched onto:
  * RandomResizedCrop(0.7-1.0)  resolution + fixed-box/detected-box framing jitter
  * RandomRotation(5)           sub-pixel tile misalignment only (portraits are axis-aligned)
  * ColorJitter(.5,.5,.5,.5)    decorrelate the red<->blue P1/P2 tint so colour can't be the cue
  * RandomGrayscale(0.2)        force some colour-free batches -> learn the weapon/shape
  * NO horizontal flip          HUD portraits never mirror in game

It reports the two deployment-critical error rates beyond accuracy:
  * m_char -> not_m_char   real m_char rejected (tolerable: pipeline keeps the detector's generic label)
  * not_m_char -> m_char   non-m_char hallucinated as an m_char (worst case: watched closely)
then trains a final model on ALL data and saves the
`{state_dict, classes, img_size, arch}` checkpoint that `vodparse.m_char.load_m_char_classifier`
expects, carrying the `display_names` map (class label -> the character name the pipeline
reports) forward from the previous checkpoint (`--names-from`, default the promoted runtime
one) so a retrain keeps the full cascade behavior: those names must agree with the character
detector's for the veto/override checks in `vodparse.m_char._refine_side` to work. Without a
previous checkpoint to copy from, the new one saves without the map and the loader falls back
to raw labels with those checks degraded (it prints a note when that happens).

Output defaults to `tool_workspace/m_char_cls_4class_clean.pt` (NON-destructive). After validating the
CV numbers, promote it to the RUNTIME LOAD PATH: the pipeline loads the m_char classifier from
`weights/m_char.pt` (M_CHAR_CLASSIFIER_PATH in src/vodparse/__init__.py; `weights/` is gitignored and
ships as a GitHub Release asset named `m_char.pt`):
    cp tool_workspace/m_char_cls_4class_clean.pt weights/m_char.pt

Reproduce (run from repo root; train with MPS/CUDA for speed if available):
    python tools/data_prep/train_m_char_4class.py --dataset tool_workspace/m_char_frames/dataset_4class_clean
    python tools/data_prep/train_m_char_4class.py --folds 0 --epochs 18   # skip CV, just train the model
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np

CLASSES = ["SF", "BR", "GN", "not_m_char"]
M_CHAR = (0, 1, 2)
DS_DEFAULT = os.path.join("tool_workspace", "m_char_frames", "dataset_4class_clean")
OUT_DEFAULT = os.path.join("tool_workspace", "m_char_cls_4class_clean.pt")
# Where display_names is carried forward from by default: the promoted runtime checkpoint
# (M_CHAR_CLASSIFIER_PATH in src/vodparse/__init__.py; ships as a GitHub Release asset).
NAMES_FROM_DEFAULT = os.path.join("weights", "m_char.pt")


def carried_display_names(prior_path):
    """The display_names map from a prior checkpoint, restricted to the POSITIVE labels this
    run trains (the negative class must never gain a display name, see
    vodparse.m_char.load_m_char_classifier), or None when the checkpoint or its map is
    missing. Carrying the map forward keeps a retrain reporting the same character names as
    the character detector, which vodparse.m_char's veto/override checks depend on; the
    names deliberately live only in checkpoints, never in this source tree."""
    import torch
    if not os.path.exists(prior_path):
        return None
    names = torch.load(prior_path, map_location="cpu").get("display_names") or {}
    positives = [CLASSES[i] for i in M_CHAR]
    kept = {c: names[c] for c in positives if c in names}
    if kept and len(kept) < len(positives):
        print(f"warning: {prior_path} carries display_names for only {sorted(kept)}; "
              f"the uncovered m_char labels will be reported raw")
    return kept or None


def load_groups(dataset):
    """filename -> disjoint-CV group, from groups.csv if present (else every file is its own group)."""
    path = os.path.join(dataset, "groups.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {r["filename"]: r["group"] for r in csv.DictReader(fh)}


def build_model(device):
    import torch.nn as nn
    import torchvision.models as M
    m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.layer4.parameters():
        p.requires_grad = True
    m.fc = nn.Linear(512, len(CLASSES))
    return m.to(device)


def make_transforms(img):
    import torchvision.transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([T.ToPILImage(), T.RandomResizedCrop(img, scale=(0.7, 1.0)),
                          T.RandomRotation(5), T.ColorJitter(0.5, 0.5, 0.5, 0.5),
                          T.RandomGrayscale(p=0.2), T.ToTensor(), norm])
    val_tf = T.Compose([T.ToPILImage(), T.Resize((img, img)), T.ToTensor(), norm])
    return train_tf, val_tf


def train_loop(model, imgs, y, idx, train_tf, cw, epochs, bs, lr, device, tag=""):
    import torch
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = torch.nn.CrossEntropyLoss(weight=cw.to(device))
    for e in range(epochs):
        order = np.random.permutation(idx)
        run, nb = 0.0, 0
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            xb = torch.stack([train_tf(imgs[i]) for i in b]).to(device)
            yb = torch.tensor(y[b]).to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        if tag:  # live per-epoch progress (loss should fall); flush so it shows in real time
            print(f"    {tag}  epoch {e + 1:>2}/{epochs}  loss {run / max(1, nb):.3f}", flush=True)


def assign_folds(files, y, groups_map, nfold, rng):
    """Disjoint folds: positives round-robin by GROUP within class (a group stays in one fold);
    negatives spread per-image. Mirrors the original player-disjoint scheme."""
    group_of = [groups_map.get(os.path.basename(f), f"neg:{os.path.basename(f)}") for f in files]
    is_pos = [not g.startswith("neg:") for g in group_of]
    folds = np.empty(len(files), int)
    for ci in M_CHAR:
        gs = sorted({group_of[i] for i in range(len(files)) if is_pos[i] and y[i] == ci})
        fold = {g: k % nfold for k, g in enumerate(gs)}
        for i in range(len(files)):
            if is_pos[i] and y[i] == ci:
                folds[i] = fold[group_of[i]]
    neg = [i for i in range(len(files)) if not is_pos[i]]
    for k, i in enumerate(rng.permutation(neg)):
        folds[i] = k % nfold
    n_groups = len({g for g in group_of if not g.startswith("neg:")})
    return folds, n_groups


def report(y, preds):
    K = len(CLASSES)
    conf = np.zeros((K, K), int)
    for t, p in zip(y, preds):
        if p >= 0:
            conf[t, p] += 1
    print(f"\noverall accuracy: {(preds == y).mean():.1%}")
    for ci, c in enumerate(CLASSES):
        tot = conf[ci].sum()
        print(f"  {c:>7} recall: {conf[ci, ci]}/{tot} = {conf[ci, ci] / max(1, tot):.0%}")
    print("confusion (rows=true, cols=pred)", CLASSES)
    for ci, c in enumerate(CLASSES):
        print(f"  {c:>7}: {conf[ci].tolist()}")
    true_mc = np.isin(y, M_CHAR)
    nm = CLASSES.index("not_m_char")
    mc_to_nm = int(((preds == nm) & true_mc).sum())
    nm_to_mc = int((np.isin(preds, M_CHAR) & (y == nm)).sum())
    recog = true_mc & np.isin(preds, M_CHAR)
    type_acc = (preds[recog] == y[recog]).mean() if recog.sum() else 0.0
    print("\n--- deployment-critical rates ---")
    print(f"  m_char -> not_m_char (real m_char rejected):    {mc_to_nm}/{int(true_mc.sum())} "
          f"= {mc_to_nm / max(1, int(true_mc.sum())):.1%}")
    print(f"  not_m_char -> m_char (hallucinated):         {nm_to_mc}/{int((y == nm).sum())} "
          f"= {nm_to_mc / max(1, int((y == nm).sum())):.1%}  <- keep low")
    print(f"  3-way type acc | recognized as an m_char:  {type_acc:.1%} (n={int(recog.sum())})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DS_DEFAULT, help=f"ImageFolder dataset (default {DS_DEFAULT})")
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help=f"Checkpoint output; NON-destructive, promote to models/ after CV "
                         f"(default {OUT_DEFAULT})")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--folds", type=int, default=5, help="CV folds for the report; 0 skips CV")
    ap.add_argument("--names-from", default=NAMES_FROM_DEFAULT,
                    help=f"prior checkpoint to carry display_names forward from; '' skips "
                         f"(default {NAMES_FROM_DEFAULT})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="torch device (default: auto cuda/mps/cpu)")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    # Resolve the carried display_names UP FRONT: a wrong/unreadable --names-from file
    # should fail here, before hours of training, not at save time.
    display_names = carried_display_names(args.names_from) if args.names_from else None
    if display_names:
        print(f"display_names will be carried forward from {args.names_from}")
    elif args.names_from:
        print(f"no display_names to carry forward from {args.names_from} (missing "
              "checkpoint or no metadata there): the saved model will report raw labels "
              "and the pipeline's detector-name checks run degraded (see "
              "vodparse.m_char.load_m_char_classifier)")
    else:
        print("--names-from '': skipping the display_names carry-forward")

    files, y = [], []
    for ci, c in enumerate(CLASSES):
        for f in sorted(glob.glob(os.path.join(glob.escape(os.path.join(args.dataset, c)), "*.png"))):
            files.append(f)
            y.append(ci)
    if not files:
        raise SystemExit(f"no images under {args.dataset}/{{{','.join(CLASSES)}}}/")
    y = np.array(y)
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
    counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    groups_map = load_groups(args.dataset)
    train_tf, val_tf = make_transforms(args.img)
    freq = np.array([counts[c] for c in CLASSES], float)
    cw = torch.tensor(freq.sum() / (len(CLASSES) * freq), dtype=torch.float32)
    print(f"device {device} | {len(files)} imgs | per-class {counts}")
    print(f"class weights: {[round(float(w), 2) for w in cw]}"
          f"{' | groups.csv missing -> per-image CV' if not groups_map else ''}")

    if args.folds and args.folds > 1:
        folds, n_groups = assign_folds(files, y, groups_map, args.folds, rng)
        print(f"player-disjoint {args.folds}-fold CV ({n_groups} unique m_char groups) ...")
        preds = np.full(len(files), -1)
        for k in range(args.folds):
            tr = np.where(folds != k)[0]
            va = np.where(folds == k)[0]
            m = build_model(device)
            train_loop(m, imgs, y, tr, train_tf, cw, args.epochs, args.batch, args.lr, device,
                       tag=f"fold {k + 1}/{args.folds}")
            m.eval()
            with torch.no_grad():
                for i in va:
                    preds[i] = int(m(val_tf(imgs[i]).unsqueeze(0).to(device)).argmax(1))
            print(f"  fold {k + 1}/{args.folds} done (val {len(va)})")
        print(f"\n=== {args.folds}-fold player-disjoint CV (n={len(files)}) ===")
        report(y, preds)

    print("\ntraining final model on ALL data ...")
    final = build_model(device)
    train_loop(final, imgs, y, np.arange(len(files)), train_tf, cw, args.epochs, args.batch,
               args.lr, device, tag="final")
    final.eval()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ckpt = {"state_dict": {k: v.cpu() for k, v in final.state_dict().items()},
            "classes": CLASSES, "img_size": args.img, "arch": "resnet18"}
    if display_names:   # resolved up front, right after arg parsing
        ckpt["display_names"] = display_names
    torch.save(ckpt, args.out)
    print(f"saved {args.out}")
    print(f"  -> validate the CV rates, then promote to the runtime load path:  "
          f"cp {args.out} weights/m_char.pt")


if __name__ == "__main__":
    main()
