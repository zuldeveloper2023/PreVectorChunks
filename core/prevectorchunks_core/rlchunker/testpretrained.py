from rlchunker import load_pretrained, chunk_text

policy = load_pretrained(device="cpu")
text = "This is sentence one. This is sentence two. And another one here. Finally, this ends."
chunks = chunk_text(text, policy)
print(chunks)
