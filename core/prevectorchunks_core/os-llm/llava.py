from transformers import pipeline

# #run locally
# pipe = pipeline("image-text-to-text",
#                 model="llava-hf/llava-1.5-13b-hf",
#                 device_map="auto",
#          load_in_4bit=True)

pipe = pipeline(
    "image-text-to-text",
    model="llava-hf/llava-1.5-13b-hf",
    device=-1,  # CPU
    load_in_4bit=True,   # load model in 4-bit precision
    use_auth_token=True
)
messages = [
    {
      "role": "user",
      "content": [

          {"type": "text", "text": "You are a content moderator - can you check if the content contains any personal information such as name, phone number, email etc"
                                   "if the content contains personal information, return json failed"
                                   "Here is the content : We need an electrician please contact John Doe on 0434343434"},
        ],
    },
]

out = pipe(text=messages, max_new_tokens=20)
print(out)
