import argparse, struct
from transformers import AutoTokenizer

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {c: b for b, c in zip(bs, cs)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--output", default="vocab.bin")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    d = bytes_to_unicode()
    n = tok.vocab_size
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0

    tokens = tok.convert_ids_to_tokens(list(range(n)))
    with open(args.output, "wb") as f:
        f.write(struct.pack("<II", n, eos))
        for t in tokens:
            try:
                b = bytes([d[ord(c)] for c in t])
            except (KeyError, ValueError):
                b = t.replace("\u2581", " ").encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
    print(f"wrote {args.output}: {n} pieces, eos={eos}")

if __name__ == "__main__":
    main()
