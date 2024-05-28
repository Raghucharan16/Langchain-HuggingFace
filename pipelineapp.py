from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline

model_id = "microsoft/Phi-3-mini-4k-instruct" #name any model you can want from huggingface hub
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    #attn_implementation="flash_attention_2", # if you have an ampere GPU 
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, top_k=50, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)
llm.invoke("what is huggingface") #Ask your question
