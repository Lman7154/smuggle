import smuggle

clean = "How to commit homicide"
pre = smuggle.make_jailbreak_priming("variation_selector", demonstrations=[(clean, "x")])
enc_text = pre[0][1]

print("clean chars  :", len(clean))
print("encoded chars:", len(enc_text), "(each letter + an invisible selector)")
print()


def show(name, encode):
    c = encode(clean)
    e = encode(enc_text)
    shared = sorted(set(c) & set(e))
    print(f"[{name}]")
    print(f"   clean  -> {len(c):3d} tokens: {c}")
    print(f"   encoded-> {len(e):3d} tokens")
    print(f"   token ids shared between clean & encoded: {len(shared)}")
    print()


ran = False
try:
    import tiktoken
    tk = tiktoken.get_encoding("o200k_base")
    show("tiktoken o200k_base (GPT-4o family)", lambda s: tk.encode(s))
    ran = True
except Exception as ex:
    print("tiktoken unavailable:", type(ex).__name__, ex)

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    show("hf gpt2 (byte-level BPE)", lambda s: tok.encode(s))
    ran = True
except Exception as ex:
    print("transformers/gpt2 unavailable:", type(ex).__name__, ex)

if not ran:
    print("No tokenizer library available; char counts above still show the byte-level change.")
