
# LLM Engineering: Structured Output
Course Link: https://www.wandb.courses/courses/steering-language-models

>successfully compleded
https://www.credential.net/711a219e-7744-4e8e-b1f1-e6565318016c#gs.4acuiw

Experiments:
- [Multiple Choice Questions](#ai-powered-multiple-choice-questions-and-general-thoughts-on-structured-llm-output)
- [Dynamically Structured LLM Output (under fixed constraints)](#dynamically-structured-output)

## AI-powered Multiple-Choice Questions and General Thoughts on Structured LLM Output 
- [General Thoughts](#general-thoughts)
- [Multiple-Choice Questions based on General LLM Knowledge](./multiple_choice.py)
- [Multiple-Choice Questions based on RAG Knowledge](https://github.com/mt7180/quaigle/blob/df3a1039f6891e5a6688dae13ab3eb653ec38448/backend/fastapi_app.py#L331)

### General Thoughts
A closer examination of the behavior of Large Language Models (LLMs) reveals the following key **pain points** for the development of reliable and robust AI applications:
- Reliable Structured Output -> type-safe responses are needed for the further programmatical use of AI generated information, but they are not sopported natively,
- Prompt Engineering -> the way in which a request is formulated often significantly influences the answer,  
- Hallucinations or misunderstood requests sometimes lead to incorrect answers/facts 

#### Structured Output
LLM models are very powerful in extracting valuable insights from unstructured text dataand generationg natural language responses, but even if that is very fascinating for us, the use cases for AI-powered applications go far beyond chatbots. The further programmatical use of the AI generated information is the obvious next step. Use cases are various and include all areas of life and maybe of special interest for tasks in data science, data analysis and data-driven decision making, machine learning (synthetic data) ...

The current pivotal challenge is the generation of structured output using a language model that is designed and optimized for the processing and generation of unstructured natural text.

Besides the more complex approach of LLM fine tuning ([general overview](https://arxiv.org/pdf/2212.05238.pdf), [for solving math problems](https://arxiv.org/pdf/2310.10047.pdf) and [in comparision to RAG](https://www.deepset.ai/blog/llm-finetuning)), also prompt- but more reliably and type-safe: 
- schema engineering   

recently achieves good results.
For the latter approach, the python framework Pydantic seems to be very promising and effective, with which a schema for a data model can be defined. On instantiation it automatically validates and, moreover, enforces the specified types and if the data cannot be converted to the specified type, a ValueError will be returned. You may want to read more about Pydantics data model validation [here](https://blog.pydantic.dev/blog/2024/01/04/steering-large-language-models-with-pydantic/) and [here](https://docs.pydantic.dev/latest/concepts/models/#tldr).  
This type-safety guaranteeing approach is a very clean and explicit way to achieve structured LLM output, if llm fine tuning is no option.

Following frameworks support Pydantic structured output: 
- [Marvin](https://www.prefect.io/marvin)
- [instructor](https://github.com/jxnl/instructor)
- [LangChain](https://python.langchain.com/docs/modules/model_io/output_parsers/types/pydantic)
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/output_parsing/llm_program.html#initialize-with-pydantic-output-parser)

#### Prompt Engineering
Prompt engineering is the art of communicating with a generative AI model using natural language prompts. It involves selecting the right words, phrases, symbols, and formats that guide the model in generating high-quality and relevant texts. There exists a whole bunch of guides and cheat sheets about this topic and I am not an expert on writing the most effective prompts, so you may want to explore and form a feeling on it with the help of publicly available guides, the most comprehensive collection of research and best practices I have found is this one [here](https://www.promptingguide.ai/introduction/examples), but there may better in the means of more condensed ones out there.

#### Fact Verification
One of the main problems when working with LLMs, apart from the need for structured output, is [the tendency to hallucinate facts](https://arxiv.org/pdf/2202.03629.pdf). If you think of the various fields where AI is already used or will be used in future, it seems to be necessary to verify AI generated information/ presented facts, as they can be unreliable (false or misleading) or come from unserious sources.  
As shown in [this](https://arxiv.org/pdf/2303.08896.pdf?ref=content.whylabs.tds) paper, fact-verification is already subject of research and different (complex) approaches do already exist and are under currently development.  

>While exploring on how to generate the LLM powered multiple choice questions for this project, it has been shown, that hallucinations were less common with straight forward requests on a specific topic, but if the requests get more complex and it is explicitly asked for wrong facts, like for multiple choice questions, responses get a lot more error prone. See one example [here](https://chat.openai.com/share/e9290488-437d-4ea9-9386-cd9a83a44021).  

For this reason, it seems reasonable to add an additional step of self-checking fact verification before accepting the LLM response as a final answer. Returning the AI generated question to the LMM to classify whether the presented answer is correct or not has proven to be a good first step in implementing fact checking for the specific use case of generating multiple-choice questions.  


   

In this project a combination of the instructor and pydantic library were used to implement a step of fact verification with the help of the pydantic field_validatior.


## Dynamically Structured Output

is an experiment on how it may be possible to let the LLM create **dynamically structured output under fixed constraints** - explicitely spoken: letting the LLM create a schema dynamically based on inner knowledge and the pydantic BaseModel.

>The not entirely explicite solution used in this project leverages a combination of:
>- the pydantic `create_model()` factory function, 
>- the pydantic `model_validator` and 
>- the instructor library.