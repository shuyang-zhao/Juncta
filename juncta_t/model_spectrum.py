"""Stage 1: peak-level transformer with cross-attention between the peaks of a spectrum and its top-k candidates, and
its training and scoring. Candidates are seen as fragment ladders, match channels and residue classes, not residues.
usage: python -m juncta_t.model_spectrum train <g> <h> | score <g> <h>   (one GPU; reruns are not bit-identical)"""
import glob
import itertools
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from juncta.peptide import AA
from juncta_t.common import CONFIG, H2O, OPTIONB_COHORTS, OX, PROTON, RESULTS, SEARCH_MASS, TMT, TOKENS

TWO_PI = 2 * math.pi
MODEL, TRAINING = CONFIG["spectrum_model"], CONFIG["training"]
ARRAYS = ("mz", "it", "prec", "tok", "cmask", "label")
MASSES = {**SEARCH_MASS, "m": AA["M"] + OX}
TOKEN_MASS = torch.tensor([MASSES.get(token, 0.0) for token in TOKENS], dtype=torch.float64)
# fragmentation-relevant residue classes: 0 proline, 1 acidic, 2 basic, 3 glycine, 4 other, 5 no residue
RESIDUE_CLASS = {**{a: 4 for a in "ACFILMNQSTVWYm"}, "P": 0, "D": 1, "E": 1, "H": 2, "K": 2, "R": 2, "G": 3}
TOKEN_CLASS = torch.tensor([RESIDUE_CLASS.get(token, 5) for token in TOKENS])


def ladder_from_tokens(tok):
    """Theoretical b / y m/z in float64: tok [..., L] -> ladder [..., L - 1, 4] (b z1, y z1, b z2, y z2), site mask."""
    cumulative = TOKEN_MASS.to(tok.device)[tok].cumsum(-1)
    length = (tok > 0).sum(-1)
    total = cumulative.gather(-1, (length - 1).clamp(min=0).unsqueeze(-1))
    prefix = cumulative[..., :-1] + TMT
    suffix = total + H2O - cumulative[..., :-1]
    columns = []
    for z in MODEL["ion_charges"]:
        columns += [(prefix + z * PROTON) / z, (suffix + z * PROTON) / z]
    site = torch.arange(tok.shape[-1] - 1, device=tok.device) < (length - 1).unsqueeze(-1)
    return (torch.stack(columns, -1) * site.unsqueeze(-1)).float(), site


class Sinusoid(nn.Module):
    """Multi-scale sinusoidal encoding; phase = (x * 2 pi / lambda) mod 2 pi in float64, reduced before any cast."""

    def __init__(self, dim, lam_min, lam_max):
        super().__init__()
        lam = torch.logspace(math.log10(lam_min), math.log10(lam_max), dim // 2, dtype=torch.float64)
        self.register_buffer("freq", TWO_PI / lam, persistent=False)

    def forward(self, x):
        phase = torch.remainder(x.double().unsqueeze(-1) * self.freq, TWO_PI)
        return torch.cat([torch.sin(phase), torch.cos(phase)], -1).float()


def match_channels(ladder, site_mask, mz, inten, peak_mask, tol_ppm):
    """Nearest observed peak of every theoretical ion within tol: [matched, signed ppm / tol, sqrt-intensity]."""
    B, K, S, I = ladder.shape
    peaks = torch.where(peak_mask, mz, torch.full_like(mz, 1e9))
    query = ladder.reshape(B, -1).contiguous()
    position = torch.searchsorted(peaks.contiguous(), query)
    best_error = torch.full_like(query, float("inf"))
    best_intensity = torch.zeros_like(query)
    for offset in (-1, 0):
        j = (position + offset).clamp(0, peaks.shape[1] - 1)
        error = (torch.gather(peaks, 1, j) - query) / query.clamp_min(1.0) * 1e6
        better = error.abs() < best_error.abs()
        best_error = torch.where(better, error, best_error)
        best_intensity = torch.where(better, torch.gather(inten, 1, j), best_intensity)
    hit = (best_error.abs() <= tol_ppm) & (query > 0)
    zero = torch.zeros_like(query)
    channels = [hit.float(), torch.where(hit, best_error / tol_ppm, zero), torch.where(hit, best_intensity, zero)]
    return torch.stack(channels, -1).reshape(B, K, S, I * 3) * site_mask.unsqueeze(-1)


def encoder_stack(layers):
    # Pre-LN stacks need the final LayerNorm.
    layer = nn.TransformerEncoderLayer(MODEL["d_model"], MODEL["n_heads"], MODEL["ff"], MODEL["dropout"],
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(MODEL["d_model"]), enable_nested_tensor=False)


class PeakEncoder(nn.Module):
    def __init__(self, d, emb, max_charge=8):
        super().__init__()
        self.mz = Sinusoid(d, *MODEL["mz_lambda"])
        self.mz_lin = nn.Linear(d, d)
        self.int_mlp = nn.Sequential(nn.Linear(1, d), nn.ReLU(), nn.Linear(d, d))
        self.prec_mass = Sinusoid(d, 0.001, 10000.0)
        self.prec_lin = nn.Linear(d, d)
        self.charge = nn.Embedding(max_charge + 1, d)
        self.rt = nn.Linear(1, d)
        self.enc = encoder_stack(MODEL["enc_layers"])
        self.proj = nn.Linear(d, emb)

    def forward(self, mz, inten, peak_mask, prec):
        x = self.mz_lin(self.mz(mz)) + self.int_mlp(inten.unsqueeze(-1))
        charge = prec[:, 1].long().clamp(0, self.charge.num_embeddings - 1)
        p = self.prec_lin(self.prec_mass(prec[:, 0])) + self.charge(charge) + self.rt(prec[:, 2:3])
        h = torch.cat([p.unsqueeze(1), x], 1)
        pad = torch.cat([torch.zeros_like(peak_mask[:, :1]), ~peak_mask], 1)
        H = self.enc(h, src_key_padding_mask=pad)
        return H, pad, F.normalize(self.proj(H[:, 0]), dim=-1)


class CandidateEncoder(nn.Module):
    """Site tokens: ion m/z, site position, flanking residue classes, match channels; <cand> holds the length."""

    def __init__(self, d, emb, n_rcls=5):
        super().__init__()
        n_ion, max_len = 2 * len(MODEL["ion_charges"]), MODEL["max_len"]
        self.ion_enc = Sinusoid(d, *MODEL["mz_lambda"])
        self.ion_lin = nn.ModuleList([nn.Linear(d, d) for _ in range(n_ion)])
        self.pos = nn.Embedding(max_len, d)
        self.rcls_l = nn.Embedding(n_rcls + 1, d)
        self.rcls_r = nn.Embedding(n_rcls + 1, d)
        self.match = nn.Linear(3 * n_ion, d)
        self.cand = nn.Parameter(torch.zeros(1, 1, d))
        self.length = nn.Embedding(max_len + 1, d)
        decoder_layer = nn.TransformerDecoderLayer(d, MODEL["n_heads"], MODEL["ff"], MODEL["dropout"], batch_first=True,
                                                   norm_first=True)
        self.dec = nn.TransformerDecoder(decoder_layer, MODEL["dec_layers"], norm=nn.LayerNorm(d))
        self.pep_enc = encoder_stack(2)
        self.pep_proj = nn.Linear(d, emb)

    def forward(self, H, pad, ladder, site_mask, cand_rcls, match):
        B, K, S, I = ladder.shape
        base = sum(self.ion_lin[i](self.ion_enc(ladder[..., i])) for i in range(I))
        base = base + self.pos.weight[:S] + self.rcls_l(cand_rcls[:, :, :S]) + self.rcls_r(cand_rcls[:, :, 1:S + 1])
        length = site_mask.sum(-1) + 1
        c = self.cand.expand(B, K, -1) + self.length(length.clamp(max=self.length.num_embeddings - 1))
        # peptide-only path (no spectrum, no match channels) for z_p
        peptide_tokens = torch.cat([c.unsqueeze(2), base], 2).reshape(B * K, S + 1, -1)
        peptide_mask = torch.cat([torch.zeros_like(site_mask[..., :1]), ~site_mask], 2).reshape(B * K, S + 1)
        peptide_mask[:, 0] = False
        z_p = self.pep_proj(self.pep_enc(peptide_tokens, src_key_padding_mask=peptide_mask)[:, 0])
        z_p = F.normalize(z_p, dim=-1).reshape(B, K, -1)
        # spectrum-conditioned path
        site_tokens = torch.cat([c.unsqueeze(2), base + self.match(match)], 2).reshape(B * K, S + 1, -1)
        out = self.dec(site_tokens, H.repeat_interleave(K, 0), tgt_key_padding_mask=peptide_mask,
                       memory_key_padding_mask=pad.repeat_interleave(K, 0))
        return out[:, 0].reshape(B, K, -1), z_p


class SetHead(nn.Module):
    """Candidates compare with each other (near-neighbour competition) plus a learned none token."""

    def __init__(self, d):
        super().__init__()
        self.none = nn.Parameter(torch.zeros(1, 1, d))
        self.enc = encoder_stack(MODEL["set_layers"])
        self.out = nn.Linear(d, 1)

    def forward(self, h, cand_mask):
        x = torch.cat([h, self.none.expand(h.shape[0], 1, -1)], 1)
        pad = torch.cat([~cand_mask, torch.zeros_like(cand_mask[:, :1])], 1)
        return self.out(self.enc(x, src_key_padding_mask=pad)).squeeze(-1).masked_fill(pad, -1e4)


class SpectrumCandidateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PeakEncoder(MODEL["d_model"], MODEL["emb_dim"])
        self.cand = CandidateEncoder(MODEL["d_model"], MODEL["emb_dim"])
        self.head = SetHead(MODEL["d_model"])
        # Unused buffers, kept so that the stored checkpoints load strictly.
        self.register_buffer("feat_mu", torch.zeros(1))
        self.register_buffer("feat_sd", torch.ones(1))

    def forward(self, b):
        H, pad, z_s = self.enc(b["mz"], b["inten"], b["peak_mask"], b["prec"])
        match = match_channels(b["ladder"], b["site_mask"], b["mz"], b["inten"], b["peak_mask"], MODEL["frag_tol_ppm"])
        h, z_p = self.cand(H, pad, b["ladder"], b["site_mask"], b["cand_rcls"], match)
        return self.head(h, b["cand_mask"]), z_s, z_p


def total_loss(logits, z_s, z_p, b):
    """Softmax CE over the K candidates + none, and InfoNCE of z_s against z_p with the whole batch as negatives."""
    label = b["label"]
    positive = (label >= 0) & (label < logits.shape[1] - 1)
    B, K, E = z_p.shape
    similarity = z_s[positive] @ z_p.reshape(B * K, E).T / MODEL["temperature"]
    similarity = similarity.masked_fill(~b["cand_mask"].reshape(-1).unsqueeze(0), -1e4)
    contrastive = F.cross_entropy(similarity, torch.nonzero(positive).squeeze(1) * K + label[positive])
    weights = TRAINING["loss_weights"]
    return weights["cand"] * F.cross_entropy(logits, label, ignore_index=-1) + weights["contrastive"] * contrastive


def load_shards(directories):
    return {name: np.concatenate([np.load(f"{d}/{name}.npy") for d in directories]) for name in ARRAYS}


def to_batch(S, idx, device, rng=None):
    """Model inputs of the rows idx. With rng the candidates go in random order and the label follows its candidate."""
    b = {name: torch.from_numpy(np.ascontiguousarray(S[name][idx])) for name in ARRAYS}
    if rng is not None:
        B, K = b["cmask"].shape
        perm = torch.from_numpy(np.argsort(rng.random((B, K)), 1))
        b["tok"] = torch.gather(b["tok"], 1, perm.view(B, K, 1).expand_as(b["tok"]))
        b["cmask"] = torch.gather(b["cmask"], 1, perm)
        label = b["label"].long()
        labelled = (label >= 0) & (label < K)
        label[labelled] = torch.argsort(perm, 1)[labelled, label[labelled]]
        b["label"] = label
    b = {name: value.to(device, non_blocking=True) for name, value in b.items()}
    tok = b["tok"].long()
    ladder, site = ladder_from_tokens(tok)
    return dict(mz=b["mz"], inten=b["it"].float(), peak_mask=b["mz"] > 0, prec=b["prec"], ladder=ladder,
                cand_rcls=TOKEN_CLASS.to(device)[tok], site_mask=site & b["cmask"].unsqueeze(-1), cand_mask=b["cmask"],
                label=b["label"].long())


def synth_none(b, generator):
    """Delete the true candidate of a random fraction of the labelled spectra and relabel them none."""
    K = b["cand_mask"].shape[1]
    label = b["label"]
    drawn = torch.rand(label.shape, device=label.device, generator=generator) < TRAINING["p_none"]
    rows = torch.nonzero((label >= 0) & (label < K) & drawn).squeeze(1)
    cand_mask = b["cand_mask"].clone()
    cand_mask[rows, label[rows]] = False
    b["cand_mask"] = cand_mask
    b["site_mask"] = b["site_mask"] & cand_mask.unsqueeze(-1)
    b["label"] = label.clone().index_fill(0, rows, K)
    return b


def train(shards, out, phase, init=None):
    """Labels come from the original search; junk spectra are labelled none. A fixed 2 % of the spectra stays out."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = TRAINING[phase]["lr"]
    S = load_shards(shards)
    n = len(S["label"])
    batch = TRAINING["batch"]
    steps = math.ceil(TRAINING[phase]["epochs"] * n / batch)
    row_hash = (np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) % np.uint64(100)
    train_idx = np.flatnonzero((row_hash >= 2) & (S["label"] >= 0))
    torch.manual_seed(TRAINING["seed"])
    model = SpectrumCandidateModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=TRAINING["weight_decay"],
                                  betas=tuple(TRAINING["adam_betas"]))
    if init:   # fine-tuning: pretrained weights, fresh optimizer and schedule
        model.load_state_dict(torch.load(init, map_location=device, weights_only=True)["model"])
    rng = np.random.default_rng(7)
    generator = torch.Generator(device=device)
    generator.manual_seed(11)
    warmup = min(TRAINING["warmup_steps"], max(steps // 10, 1))
    model.train()
    for step in range(steps):
        idx = np.sort(rng.choice(train_idx, batch, replace=False))
        b = synth_none(to_batch(S, idx, device, rng), generator)
        cosine = 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))) * 0.9 + 0.1
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / warmup) * cosine
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            outputs = model(b)
        loss = total_loss(*[x.float() for x in outputs], b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["grad_clip"])
        optimizer.step()
    os.makedirs(out, exist_ok=True)
    torch.save(dict(model=model.state_dict()), f"{out}/ckpt.pt")


@torch.no_grad()
def score(shard, checkpoint, out, batch=512):
    """scores.npy [n, K], logits.npy [n, K + 1], in fp32: bf16 logits turn the top of the score list into ties."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpectrumCandidateModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.eval()
    S = load_shards([shard])
    n, K = S["cmask"].shape
    scores = np.full((n, K), -np.inf, np.float32)
    logits = np.zeros((n, K + 1), np.float32)
    for start in range(0, n, batch):
        idx = np.arange(start, min(start + batch, n))
        b = to_batch(S, idx, device)
        batch_logits = model(b)[0].float()
        logits[idx] = batch_logits.cpu().numpy()
        log_odds = batch_logits.double()   # l_k - logsumexp(l_{j != k}) over the set, including none
        for k in range(K):
            others = log_odds.clone()
            others[:, k] = float("-inf")
            scores[idx, k] = (log_odds[:, k] - torch.logsumexp(others, -1)).cpu().numpy()
        scores[idx] = np.where(b["cand_mask"].cpu().numpy(), scores[idx], -np.inf)
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/scores.npy", scores)
    np.save(f"{out}/logits.npy", logits)


if __name__ == "__main__":
    mode, g, h = sys.argv[1], *map(int, sys.argv[2:])
    run, shards = f"{RESULTS}/runs/s1_g{g}h{h}", f"{RESULTS}/shards_v3"
    if mode == "train":   # S_{g,h} sees genes of coarse group g and plex half h only
        train([f"{shards}/{c}_pre_g{g}h{h}" for c in OPTIONB_COHORTS], f"{RESULTS}/runs/pre_g{g}h{h}", "pretrain")
        train([f"{shards}/{c}_f{f}_train_h{h}" for f in range(g, 6, 2) for c in OPTIONB_COHORTS], run, "finetune",
              f"{RESULTS}/runs/pre_g{g}h{h}/ckpt.pt")
    else:   # it scores the other group's half 1 - h; S_{0,h} also production half 1 - h
        for f, c, t in itertools.product(range(1 - g, 6, 2), OPTIONB_COHORTS, ("", "_trunc")):
            name = f"{c}_f{f}_eval_h{1 - h}{t}"
            score(f"{shards}/{name}", f"{run}/ckpt.pt", f"{run}/score_v3_{name}_fp32")
        for shard in glob.glob(f"{RESULTS}/prod/shards/*_h{1 - h}_c*") if g == 0 else []:
            score(shard, f"{run}/ckpt.pt", f"{RESULTS}/runs/prod_s1/score_prod_{os.path.basename(shard)}_fp32")
