import json

# === 入力ファイル ===
MINIMIND_PATH = "/home/ubuntu/minimind/model/minimind_tokenizer.json"
GPT2_TOKENIZER_PATH = "/home/ubuntu/minimind/model/gpt2_tokenizer.json"
OUTPUT_PATH = "/home/ubuntu/minimind/model/minimind_tokenizer_english.json"

# === minimind 読み込み ===
with open(MINIMIND_PATH, "r", encoding="utf-8") as f:
    minimind = json.load(f)

# === gpt2 tokenizer 読み込み ===
with open(GPT2_TOKENIZER_PATH, "r", encoding="utf-8") as f:
    gpt2_tokenizer = json.load(f)

# === vocabとmerges取得 ===
gpt2_vocab = gpt2_tokenizer["model"]["vocab"]
merges = gpt2_tokenizer["model"]["merges"]

# === vocab IDを +3 シフト ===
shifted_vocab = {}
for token, old_id in gpt2_vocab.items():
    shifted_vocab[token] = old_id + 3

# === special tokensを追加 ===
new_vocab = {
    "<|endoftext|>": 0,
    "<|im_start|>": 1,
    "<|im_end|>": 2,
    **shifted_vocab
}

# === minimindへ反映 ===
minimind["model"]["vocab"] = new_vocab
minimind["model"]["merges"] = merges

# === 保存 ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(minimind, f, ensure_ascii=False, indent=2)

print(f"✅ minimind_tokenizer_english.json saved to {OUTPUT_PATH}")
print(f"➡ vocab size: {len(new_vocab)} tokens (expected 50257+3-1=50259). 1 results from a duplication of <|endoftext|>")")
    