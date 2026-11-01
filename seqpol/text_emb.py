"""CLIP ViT-B/32 text embeddings for every instruction string in the decoded data."""
import glob, json
import numpy as np, torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

texts = set()
for f in glob.glob("/data/workspaces/mikasa/data_mikasa_robo/mm/t*/meta.json"):
    texts |= set(json.load(open(f))["tasks"])
texts = sorted(texts)
tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
m = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").eval()
embs = []
with torch.no_grad():
    for i in range(0, len(texts), 256):
        x = tok(texts[i:i + 256], padding=True, truncation=True, max_length=77, return_tensors="pt")
        e = m(**x).text_embeds
        embs.append(torch.nn.functional.normalize(e, dim=-1).numpy())
np.savez("/data/workspaces/mikasa/data_mikasa_robo/text_emb.npz", emb=np.concatenate(embs), text=np.array(texts, dtype=object))
print(len(texts))
