from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct", # You can use any model id from the huggingface_hub
    task="text-generation", # mention the desired task eg: task generation, summarization, texttotext translation
    pipeline_kwargs={
        "max_new_tokens": 100, #there will be limit and depens on your choosen model
        "top_k": 50,
        "temperature": 0.1, #strictness
    },
)
llm.invoke("Hugging Face is")
