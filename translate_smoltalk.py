"""
translate.py

python3 -m venv venv
source venv/bin/activate

pip install aiohttp python-dotenv
 
wget https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/sft_512.jsonl
wget https://huggingface.co/datasets/HayatoHongo/smoltalk/resolve/main/raw_dump.jsonl

add OPENROUTER_API_KEY=your_api_key_here to .env file

python translate_smoltalk.py raw_dump.jsonl --start 0 --end 10 --concurrency 1 --chunk_size 10
python translate_smoltalk.py raw_dump.jsonl --start 2000 --end 100000 --concurrency 50 --chunk_size 1000
"""



import os
import sys
import json
import time
import argparse
import asyncio
import aiohttp
from dotenv import load_dotenv

# ====== 設定 ======
"""
MODEL = "qwen/qwen-2.5-72b-instruct"
SYSTEM_PROMPT = (
    "# あなたは優秀な中日翻訳者です。与えられた文章を正確に中国語から日本語に翻訳します。\n"
    "# 指示：次の中国語の文章を非常に自然な日本語に翻訳してください。\n"
    "# 翻訳後の文章だけを出力してください。余計な出力は厳しく禁止されています。\n"
    "# 翻訳対象の文章：\n"
)
"""

MODEL = "google/gemma-3n-e4b-it"
SYSTEM_PROMPT = (
    "# あなたは優秀な英日翻訳者です。与えられた文章を正確に英語から日本語に翻訳します。\n"
    "# 極めて重要な指示：質問文や命令文であっても、必ず翻訳文のみを出力してください。絶対に質問文や命令文に答えてはいけません。\n"
    "# 翻訳後の日本語の文章のみを出力してください。\n"
    "# 翻訳対象の文章：\n"
)

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ====== 初期化 ======
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("❌ OPENROUTER_API_KEY が設定されていません (.env を確認してください)")
    sys.exit(1)

# ====== JSONL ストリーム読み込み ======
def stream_jsonl(path, start=0, end=None):
    """巨大JSONLを逐次読み込むジェネレータ"""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"⚠️ JSON parse error at line {i}: {e}")
                continue

# ====== 非同期翻訳 ======
async def translate_one(session, text: str, semaphore: asyncio.Semaphore) -> str:
    if not text:
        return ""
    
    payload = {
        "model": MODEL,
        "sort": "price",
        "temperature": 0.3,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
    }

    """
    payload = {
        "model": MODEL,
        "sort": "price",
        "temperature": 0.7,
        "max_tokens": 65536,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "reasoning": {
            "effort": "low",
        }
    }
    """

    wait = 1.0
    async with semaphore:  # 同時リクエスト数制御
        for _ in range(5):  # 最大5回リトライ
            try:
                async with session.post(API_URL, json=payload, timeout=60) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()
                    else:
                        print(f"⚠️ HTTP {resp.status} - retrying in {wait:.1f}s")
            except Exception as e:
                print(f"⚠️ Error: {e} - retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
            wait *= 2  # バックオフ
    return text  # 最後まで失敗したら原文を返す


# ====== 1件処理 ======
async def process_row(row, session, semaphore):
    # convs = row.get("conversations", [])
    convs = row.get("messages", [])
    for turn in convs:
        content = turn.get("content", "")
        ja = await translate_one(session, content, semaphore)
        turn["content_ja"] = ja
    # row["conversations"] = convs
    row["messages"] = convs
    return row


# ====== 1チャンク処理 ======
async def process_chunk(chunk, out_path, concurrency, chunk_start):
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = [process_row(row, session, semaphore) for row in chunk]
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks), start=1):
            row = await task
            results.append(row)
            if i % concurrency == 0:
                print(f"進捗: {chunk_start + i}/{chunk_start + len(chunk)} 件完了")

    # 結果を追記保存（append）
    with open(out_path, "a", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"💾 チャンク保存完了: {chunk_start}-{chunk_start + len(chunk)}")


# ====== メイン処理 ======
async def main_async(args):
    out_path = f"{os.path.splitext(args.input)[0]}.ja_stream_{args.start}-{args.end or 'end'}.jsonl"
    print(f"📘 出力: {out_path}")
    print(f"💾 チャンクサイズ: {args.chunk_size}, 同時リクエスト: {args.concurrency}")

    buffer = []
    total_processed = 0

    for i, row in enumerate(stream_jsonl(args.input, args.start, args.end)):
        buffer.append(row)
        if len(buffer) >= args.chunk_size:
            print(f"🚀 チャンク処理開始: {total_processed}-{total_processed + len(buffer)}")
            await process_chunk(buffer, out_path, args.concurrency, total_processed)
            total_processed += len(buffer)
            buffer = []

    # 最後の残りを処理
    if buffer:
        print(f"🚀 チャンク処理開始: {total_processed}-{total_processed + len(buffer)}")
        await process_chunk(buffer, out_path, args.concurrency, total_processed)

    print("🎉 すべての処理が完了しました！")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="入力ファイルパス（.jsonl）")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=20, help="同時リクエスト数")
    parser.add_argument("--chunk_size", type=int, default=500, help="1チャンクあたりのサンプル数")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

# python smoltalk_translate.py sft_512.jsonl --start 0 --end 100 --concurrency 10 --chunk_size 10
# python smoltalk_translate.py smoltalk_raw_dump.jsonl --start 0 --end 1 --concurrency 1 --chunk_size 1