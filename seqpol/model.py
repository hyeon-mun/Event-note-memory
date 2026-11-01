"""Sequence-memory flow-matching policy.

Two streams:
  * reactive: trainable CNN on the current two-camera frame + proprio (fine control)
  * memory:   per-step frozen DINOv2 tokens (+ proprio) -> memory module over the whole episode
A flow-matching DiT head generates an action chunk conditioned on [reactive, task, memory].
Memory can be replaced by a learned null vector -> enables memory guidance at inference
(v = v_null + w * (v_mem - v_null)) and the paired memory-gain estimate dL = L_null - L_mem.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- reactive encoder
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.n2 = nn.GroupNorm(8, cout)
        self.sk = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(h + (x if self.sk is None else self.sk(x)))


class SpatialSoftmax(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        p = F.softmax(x.flatten(2), -1).view(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        return torch.cat([(p.sum(3) * ys).sum(2), (p.sum(2) * xs).sum(2)], 1)


class CamEncoder(nn.Module):
    """128x128 RGB -> d vector (ResNet-ish, GroupNorm, spatial softmax)."""

    def __init__(self, d=256, w=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w, 5, 2, 2, bias=False), nn.GroupNorm(8, w), nn.ReLU())
        self.body = nn.Sequential(ResBlock(w, w, 1), ResBlock(w, 2 * w, 2), ResBlock(2 * w, 4 * w, 2),
                                  ResBlock(4 * w, 8 * w, 2))           # 64->8
        self.kp = nn.Conv2d(8 * w, 64, 1)
        self.ss = SpatialSoftmax()
        self.pool_proj = nn.Linear(8 * w, d // 2)
        self.out = nn.Linear(128 + d // 2, d)

    def forward(self, x, spatial=False):  # x float (B,3,H,W) normalized
        h = self.body(self.stem(x))
        kp = self.ss(self.kp(h))
        g = self.pool_proj(h.mean((2, 3)))
        out = self.out(torch.cat([kp, g], 1))
        return (out, h) if spatial else out


_SIGLIP = {}


def siglip_vit(device):
    """Frozen SigLIP-B/16 vision tower, shared across encoders, kept out of state_dicts."""
    if device not in _SIGLIP:
        from transformers import SiglipVisionModel
        _SIGLIP[device] = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").vision_model \
            .to(device).eval().requires_grad_(False)
    return _SIGLIP[device]


class SiglipEncoder(nn.Module):
    """Frozen SigLIP patch tokens (14x14x768) -> trainable projection, attention pooling and spatial map."""

    def __init__(self, d=256):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d))
        self.pool_q = nn.Parameter(torch.randn(4, d) * 0.02)
        self.out = nn.Linear(4 * d, d)

    def forward(self, x, spatial=False):   # x float (B,3,H,W) in [0,1]
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            tok = siglip_vit(x.device)(pixel_values=(x - 0.5) / 0.5).last_hidden_state   # (B,196,768)
        tok = self.proj(tok)
        a = torch.softmax(torch.einsum("qd,bpd->bqp", self.pool_q, tok) / tok.shape[-1] ** 0.5, -1)
        out = self.out(torch.einsum("bqp,bpd->bqd", a, tok).flatten(1))
        if not spatial:
            return out
        B = tok.shape[0]
        return out, tok.transpose(1, 2).reshape(B, -1, 14, 14)


class FindReadout(nn.Module):
    """Memory-as-query top-down attention: H queries from the memory state attend over the current frame's
    spatial tokens (both cameras). Returns attended features and expected (x, y) per query/camera."""

    def __init__(self, d_mem, c=256, heads=4, d=128):
        super().__init__()
        self.h, self.d = heads, d
        self.q = nn.Linear(d_mem, heads * d)
        self.k = nn.Conv2d(c, heads * d, 1); self.v = nn.Conv2d(c, heads * d, 1)
        self.out_dim = 2 * heads * (d + 2)

    def forward(self, mem, maps):
        outs = []
        q = self.q(mem).view(-1, self.h, self.d)                               # (N,h,d)
        for fm in maps:                                                         # (N,c,H,W)
            N, _, H, W = fm.shape
            k = self.k(fm).view(N, self.h, self.d, H * W); v = self.v(fm).view(N, self.h, self.d, H * W)
            a = torch.softmax(torch.einsum("nhd,nhdp->nhp", q, k) / math.sqrt(self.d), -1)
            ys = torch.linspace(-1, 1, H, device=fm.device).repeat_interleave(W)
            xs = torch.linspace(-1, 1, W, device=fm.device).repeat(H)
            pos = torch.stack([(a * xs).sum(-1), (a * ys).sum(-1)], -1)           # (N,h,2)
            outs.append(torch.cat([torch.einsum("nhp,nhdp->nhd", a, v), pos.to(v.dtype)], -1).flatten(1))
        return torch.cat(outs, -1)


class GoalHead(nn.Module):
    """Memory -> spatial goal grounding. A full-resolution conv encoder of the current two-camera frame is
    FiLM-modulated by the memory state; logits over a 32x32 grid of the top image locate the hindsight goal
    (EE position at the next key event). Trained with cross-entropy on the projected demo goal."""

    def __init__(self, d_mem, d=64, n=32, match=False, rf=11):
        super().__init__()
        conv = lambda i, o, s: nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())
        self.match = match
        if match:
            # local keys only (receptive field ~11 px, no coordinates, top camera): the score of a location depends
            # on its appearance and the memory query alone, so the scene layout cannot be memorised as a shortcut
            c1 = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 1), nn.GELU())
            self.rf = rf
            if rf == 1:      # appearance-only keys: per-pixel colour embedding, scores max-pooled into cells
                self.enc = nn.Sequential(c1(3, d), c1(d, d), nn.Conv2d(d, d, 1))
            else:
                self.enc = nn.Sequential(conv(3, d, 1), c1(d, d), conv(d, d, 2), c1(d, d), conv(d, d, 2), nn.Conv2d(d, d, 1))
            self.q = nn.Sequential(nn.Linear(d_mem, d), nn.GELU(), nn.Linear(d, d))
            self.n = n
            return
        self.enc = nn.Sequential(conv(6, d, 1), conv(d, d, 2), conv(d, d, 1), conv(d, d, 2), conv(d, d, 1))
        self.film = nn.Linear(d_mem, 2 * d)
        self.head = nn.Sequential(conv(d + 2, d, 1), nn.Conv2d(d, 1, 1))
        self.n = n
        g = torch.linspace(-1, 1, n)
        self.register_buffer("grid", torch.stack(torch.meshgrid(g, g, indexing="ij"), 0)[None])

    def forward(self, top, wrist, mem):          # uint8 (N,H,W,3) x2, (N,d_mem) -> logits (N, n*n)
        if self.match:
            k = self.enc(top.permute(0, 3, 1, 2).float() / 255.0)
            sc = torch.einsum("nd,ndhw->nhw", self.q(mem.float()), k) / k.shape[1] ** 0.5
            if self.rf == 1:
                sc = F.max_pool2d(sc[:, None], 128 // self.n)[:, 0]
            return sc.flatten(1)
        h = self.enc(torch.cat([top, wrist], -1).permute(0, 3, 1, 2).float() / 255.0)
        g, b = self.film(mem.float()).chunk(2, -1)
        h = h * (1 + g[..., None, None]) + b[..., None, None]
        return self.head(torch.cat([h, self.grid.expand(len(h), -1, -1, -1)], 1)).flatten(1)

    def cell(self, uv):                         # pixel (N,2) -> cell index
        n = self.n
        return (uv[:, 1] * n / 128).long().clamp(0, n - 1) * n + (uv[:, 0] * n / 128).long().clamp(0, n - 1)

    def uv(self, logits):                       # argmax cell -> pixel centre (N,2)
        k = logits.argmax(-1); n = self.n
        return torch.stack([(k % n).float() + 0.5, (k // n).float() + 0.5], -1) * 128 / n


# ---------------------------------------------------------------- memory modules
def mem_in_dim(tokens):
    return 2 * tokens * 384


def gumbel_gate(logit, training, tau=1.0):
    """Hard {0,1} gate in the forward pass, sigmoid straight-through gradient. Noisy while training."""
    if training:
        u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
        logit = logit + torch.log(u) - torch.log1p(-u)
    y = torch.sigmoid(logit / tau)
    return (y > 0.5).to(y.dtype) + y - y.detach()


class FSQ(nn.Module):
    """Finite scalar quantisation (Mentzer et al. 2023): d -> len(levels) bounded scalars, rounded; straight-through."""

    def __init__(self, d, levels=(5, 5, 5)):
        super().__init__()
        self.levels = torch.tensor(levels, dtype=torch.float32)
        self.down, self.up = nn.Linear(d, len(levels)), nn.Linear(len(levels), d)

    def forward(self, x):
        L = self.levels.to(x.device)
        z = torch.tanh(self.down(x).float()) * (L - 1) / 2
        q = z + (torch.round(z) - z).detach()
        return self.up((q / ((L - 1) / 2)).to(x.dtype)), torch.round(z)


class SlotMemory(nn.Module):
    """Discrete event memory.

    write: step s is stored as a slot iff g_s = 1 (learned gate on [h_s, h_s - h_{s-1}])
    erase (todo only): slot s is deleted at the first u > s with e_{s,u} = 1, a gate on (slot key, current input)
    read : m_t = attention over live slots {s <= t : g_s * r_{s,t} = 1} plus one learned "empty" slot,
           with per-head mean log-age and log #live slots appended.
    Training runs all T steps in parallel; at inference the gates are thresholded, so the memory really is a
    short list of slots. self.aux holds the soft write rate / live-slot count for sparsity penalties.
    """

    def __init__(self, d, heads=4, erase=False, de=64, quant=None):
        super().__init__()
        self.h, self.dh, self.erase = heads, d // heads, erase
        self.fsq = FSQ(d, quant) if quant else None
        self.conv = nn.Conv1d(d, d, 3, groups=8)                    # causal local context (pad left)
        self.w_gate = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, 1))
        nn.init.constant_(self.w_gate[-1].bias, 0.0)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.empty_k = nn.Parameter(torch.zeros(d)); self.empty_v = nn.Parameter(torch.zeros(d))
        if erase:
            self.ek, self.eq = nn.Linear(d, de), nn.Linear(d, de)
            self.e_bias = nn.Parameter(torch.tensor(-7.0))
        self.out = nn.Sequential(nn.Linear(2 * d + heads + 1, d), nn.GELU(), nn.Linear(d, d))
        self.aux, self.last = {}, {}

    def forward(self, e):
        B, T, d = e.shape
        h = e + self.conv(F.pad(e.transpose(1, 2), (2, 0))).transpose(1, 2)
        dh_ = torch.cat([torch.zeros_like(h[:, :1]), h[:, 1:] - h[:, :-1]], 1)
        wl = self.w_gate(torch.cat([h, dh_], -1)).squeeze(-1).float()           # (B,T)
        g = gumbel_gate(wl, self.training)
        causal = torch.tril(torch.ones(T, T, device=e.device, dtype=torch.bool))  # [t, s]: s <= t
        G = g[:, None, :] * causal                                              # (B,t,s)
        if self.erase:
            el = torch.einsum("bsc,buc->bsu", self.ek(h).float(), self.eq(h).float()) / 8.0 + self.e_bias
            if self.training:
                u = torch.rand_like(el).clamp(1e-6, 1 - 1e-6); el = el + torch.log(u) - torch.log1p(-u)
            after = torch.triu(torch.ones(T, T, device=e.device, dtype=torch.bool), 1)  # u > s
            # survival of slot s up to t: hard (no erase event in (s,t]) forward, soft log-space surrogate backward
            hard = (torch.cumsum(((el > 0) & after).float(), 2) == 0).float()
            soft = torch.exp(torch.cumsum(F.logsigmoid(-el) * after, 2))
            R = (hard + soft - soft.detach()).transpose(1, 2)                   # (B,t,s)
            G = G * R
        # attention over slots (+ empty slot), weights multiplied by the hard live mask G
        slot = h
        if self.fsq is not None:          # slot content = discrete code ("note"), keys/values derived from it
            slot, codes = self.fsq(h); self.last["codes"] = codes.detach()
        q = self.q(h).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(slot).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(slot).view(B, T, self.h, self.dh).transpose(1, 2)
        ek = self.empty_k.view(1, self.h, 1, self.dh); ev = self.empty_v.view(1, self.h, 1, self.dh)
        sc = torch.einsum("bhtc,bhsc->bhts", q, k).float() / math.sqrt(self.dh)
        sc0 = (q * ek).sum(-1, keepdim=True).float() / math.sqrt(self.dh)        # (B,h,T,1)
        sc = sc.masked_fill(~causal, float("-inf"))                              # future slots never read
        mx = torch.maximum(sc.amax(-1, keepdim=True), sc0).detach()
        w = torch.exp(sc - mx) * G[:, None]; w0 = torch.exp(sc0 - mx)
        Z = w.sum(-1, keepdim=True) + w0
        w, w0 = w / Z, w0 / Z
        r = torch.einsum("bhts,bhsc->bhtc", w.to(v.dtype), v) + w0.to(v.dtype) * ev
        r = r.transpose(1, 2).reshape(B, T, d)
        tt = torch.arange(T, device=e.device, dtype=torch.float32)
        logage = torch.log1p((tt[:, None] - tt[None, :]).clamp(min=0))           # (t,s)
        age = (w * logage).sum(-1).transpose(1, 2)                              # (B,T,h)
        live = G.sum(-1, keepdim=True)
        m = self.out(torch.cat([r, h, age.to(r.dtype), torch.log1p(live).to(r.dtype)], -1))
        ps = torch.sigmoid(wl)
        self.aux = {"write_rate": ps.mean(), "live": (live.squeeze(-1) / T).mean() if self.erase else ps.mean()}
        self.last.update({"write": g.detach(), "live": live.detach().squeeze(-1)})
        return m


class MemoryModule(nn.Module):
    """Maps per-step memory inputs (B,T,Din) -> per-step memory state (B,T,d). Causal."""

    def __init__(self, kind, din, d=512, layers=4, heads=8, stack=8, max_len=2304, quant=None, bottleneck=None):
        super().__init__()
        self.kind = kind
        # discrete note: the memory state itself is an FSQ code (prod(levels) symbols) -> cannot index episodes
        self.bott = FSQ(d, bottleneck) if bottleneck else None
        self.inp = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, d), nn.GELU(), nn.Linear(d, d))
        if kind == "stack":
            self.stack = stack
            self.net = nn.Sequential(nn.Linear(d * stack, d), nn.GELU(), nn.Linear(d, d))
        elif kind == "gru":
            self.net = nn.GRU(d, d, num_layers=2, batch_first=True)
        elif kind == "tf":
            self.pos = nn.Parameter(torch.zeros(1, max_len, d)); nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True,
                                               activation="gelu")
            self.net = nn.TransformerEncoder(layer, layers)
        elif kind in ("event", "todo"):
            self.net = SlotMemory(d, heads=4, erase=(kind == "todo"), quant=quant)
        elif kind == "none":
            pass
        else:
            raise ValueError(kind)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, h0=None):
        """x: (B,T,Din). Returns (B,T,d), new recurrent state (gru only)."""
        e = self.inp(x)
        hN = None
        if self.kind == "none":
            m = torch.zeros_like(e)
        elif self.kind == "stack":
            B, T, d = e.shape
            pad = torch.cat([e[:, :1].expand(B, self.stack - 1, d), e], 1)
            m = self.net(pad.unfold(1, self.stack, 1).permute(0, 1, 3, 2).reshape(B, T, -1))
        elif self.kind == "gru":
            with torch.autocast("cuda", enabled=False):   # cuDNN GRU has no bf16 kernel
                m, hN = self.net(e.float(), None if h0 is None else h0.float())
        elif self.kind in ("event", "todo"):
            m = self.net(e)
        else:
            T = e.shape[1]
            mask = torch.triu(torch.full((T, T), float("-inf"), device=e.device), 1)
            m = self.net(e + self.pos[:, :T], mask=mask, is_causal=True)
        m = self.norm(m)
        if self.bott is not None:
            m, self.codes = self.bott(m)
        return m, hN


# ---------------------------------------------------------------- flow head
def timestep_emb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class DiTBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False); self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d); nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.att(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class FlowHead(nn.Module):
    def __init__(self, cond_dim, act_dim=7, horizon=8, d=256, layers=4, heads=4):
        super().__init__()
        self.a_in = nn.Linear(act_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, horizon, d)); nn.init.normal_(self.pos, std=0.02)
        self.c_in = nn.Sequential(nn.Linear(cond_dim + d, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(layers)])
        self.n_out = nn.LayerNorm(d, elementwise_affine=False)
        self.out = nn.Linear(d, act_dim); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.d = d

    def forward(self, x_t, t, cond):
        c = self.c_in(torch.cat([cond, timestep_emb(t, self.d)], -1))
        h = self.a_in(x_t) + self.pos
        for b in self.blocks:
            h = b(h, c)
        return self.out(self.n_out(h))


# ---------------------------------------------------------------- full policy
class SeqPolicy(nn.Module):
    def __init__(self, mem="gru", mem_tokens=5, d_mem=512, horizon=8, use_proprio_mem=True,
                 text_dim=512, mem_layers=4, stack=8, quant=None, mem_src="dino", find=False, vis="cnn", goal=False, goal_match=False, mem_bottleneck=None, goal_rf=11):
        super().__init__()
        self.horizon, self.mem_kind, self.mem_tokens = horizon, mem, mem_tokens
        self.use_proprio_mem = use_proprio_mem
        self.vis = vis
        Enc = SiglipEncoder if vis == "siglip" else CamEncoder
        self.enc_top, self.enc_wrist = Enc(256), Enc(256)
        self.prop = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 64))
        self.text = nn.Sequential(nn.Linear(text_dim, 128), nn.GELU(), nn.Linear(128, 128))
        self.mem_src = mem_src
        # "dino": frozen DINOv2 tokens (+proprio, text); "cnn": the (augmented, trainable) reactive embedding
        din = {"dino": mem_in_dim(mem_tokens) + (8 if use_proprio_mem else 0) + 128,
               "cnn": 256 * 2 + 64 + 128, "oracle": 16}[mem_src]   # oracle: one-hot cue label (upper bound)
        self.memory = MemoryModule(mem, din, d_mem, layers=mem_layers, stack=stack, quant=quant, bottleneck=mem_bottleneck)
        self.null_mem = nn.Parameter(torch.zeros(d_mem))
        self.find = FindReadout(d_mem) if find else None
        # goal: memory grounded into a top-image pixel, given to the action head as a Fourier-embedded point
        self.goal = GoalHead(d_mem, match=goal_match, rf=goal_rf) if goal else None
        if goal:
            self.goal_emb = nn.Sequential(nn.Linear(2 * 2 * 4, 64), nn.GELU(), nn.Linear(64, 64))
            self.null_goal = nn.Parameter(torch.zeros(64))
            self._goal_uv = None
        self.cond_dim = 256 * 2 + 64 + 128 + d_mem + (self.find.out_dim if find else 0) + (64 if goal else 0)
        self.head = FlowHead(self.cond_dim, 7, horizon)
        # per-task action scale (|a| 99th percentile per dim): flow runs in a_norm = a / a_scale[tid]
        self.register_buffer("a_scale", torch.ones(128, 7))
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # -- inputs
    def mem_inputs(self, feat, proprio, text):
        """feat (B,T,2,tokens,384) already pooled (data.pool_tokens), proprio (B,T,7), text (B,512) -> (B,T,Din)."""
        B, T = feat.shape[:2]
        x = [feat.float().reshape(B, T, -1)]
        if self.use_proprio_mem:
            x.append(proprio)
        x.append(self.text(text)[:, None].expand(B, T, -1))
        return torch.cat(x, -1)

    def reactive(self, top, wrist, proprio, text):
        """top/wrist uint8 (N,H,W,3); returns (N, 256*2+64+128)."""
        def prep(x):
            x = x.permute(0, 3, 1, 2).float() / 255.0
            return x if self.vis == "siglip" else (x - self.img_mean) / self.img_std
        et, mt = self.enc_top(prep(top), spatial=True); ew, mw = self.enc_wrist(prep(wrist), spatial=True)
        self._maps = (mt, mw)          # spatial maps of the current frame, used by the find readout
        return torch.cat([et, ew, self.prop(proprio), self.text(text)], -1)

    def cond(self, react, mem_state, use_mem):
        """use_mem: bool tensor (N,) -> null memory where False."""
        m = torch.where(use_mem[:, None], mem_state, self.null_mem.to(mem_state.dtype).expand_as(mem_state))
        parts = [react, m]
        if self.find is not None:
            parts.append(self.find(m, self._maps).to(react.dtype))
        if self.goal is not None:
            x = (self._goal_uv / 64.0 - 1.0)[..., None] * (2.0 ** torch.arange(4, device=react.device)) * math.pi
            ge = self.goal_emb(torch.cat([x.sin(), x.cos()], -1).flatten(1).to(react.dtype))
            parts.append(torch.where(use_mem[:, None], ge, self.null_goal.to(ge.dtype).expand_as(ge)))
        return torch.cat(parts, -1)

    # -- sampling
    @torch.no_grad()
    def sample(self, react, mem_state, steps=10, guidance=1.0, generator=None, tid=None):
        N = react.shape[0]
        x = torch.randn(N, self.horizon, 7, device=react.device, generator=generator)
        c_mem = self.cond(react, mem_state, torch.ones(N, dtype=torch.bool, device=react.device))
        c_null = self.cond(react, mem_state, torch.zeros(N, dtype=torch.bool, device=react.device))
        for i in range(steps):
            t = torch.full((N,), i / steps, device=react.device)
            v = self.head(x, t, c_mem)
            if guidance != 1.0:
                v0 = self.head(x, t, c_null)
                v = v0 + guidance * (v - v0)
            x = x + v / steps
        if tid is not None:
            x = x * self.a_scale[tid].view(-1, 1, 7) if torch.is_tensor(tid) else x * self.a_scale[tid]
        return x.clamp(-1, 1)
