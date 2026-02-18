from transformers import pipeline

# ----- Step 1: Load the model using a text-generation pipeline -----
# DeepSeek-R1-Distill-Qwen-1.5B is a text-only model
pipe = pipeline(
    "text-generation",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device=0  # set to -1 for CPU, or 0 for GPU if available
)

# ----- Step 2: Define your prompt -----
prompt = "Hello, are you Jon?"

# ----- Step 3: Run inference -----
out = pipe(
    prompt,
    max_new_tokens=50,  # controls length of generated output
    do_sample=True,     # optional: random sampling for variation
    temperature=0.7     # optional: controls creativity
)

# ----- Step 4: Print output -----
# `out` is a list of dicts, each dict has 'generated_text'
print("Model response:", out[0]['generated_text'])
