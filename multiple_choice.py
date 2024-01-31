# https://github.com/jxnl/instructor/blob/b7176fb0a2c88c6ec5a67b6c614da613e531167d/examples/validators/chain_of_thought_validator.py#L4
import instructor
from instructor.function_calls import OpenAISchema
from openai.types.chat import ChatCompletion

import wandb
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, AfterValidator, ValidationInfo, field_validator
from typing import Annotated, Iterable, Optional

LOG_WITH_WANDB = True
TEST_MANUALLY = False


if LOG_WITH_WANDB: run = wandb.init(project="model_validation")
load_dotenv()

client = instructor.patch(OpenAI())

class AnswerVerification(OpenAISchema):
    answer_classification: bool = Field(
        default=True,
        description="wether the answer is correct"
    )
    answer_validated: bool = Field(
        default=True,
        description="wether the answer is validated",
    )
    maybe_new_answer: Optional[str] = Field(
        default=None,
        description="new answer if the oanswer is not validated, otherwise None",
    )


class Answer(BaseModel):
    answer_text: str = Field(..., description="string representation of the answer")
    answer_classification: bool = Field(..., description="classifies if the answer is correct (True) or incorrect (False)")


class MultipleChoiceQuestion(BaseModel):
    """Data Model for a multiple choice question with three answers, while only one should be correct"""

    question: str = Field(
        ...,
        description="""An interesting and unique question related to the given topic.
        """,
    )
    correct_answer: Answer = Field(..., description="A Correct answer to the question")
    wrong_answer_1: Answer = Field(
        ..., description="a unique wrong answer to the question"
    )
    wrong_answer_2: Answer = Field(
        ...,
        description="""a unique wrong answer to the question which is different 
        from wrong_answer_1 and not an empty string
        """,
    )

    
    @field_validator('wrong_answer_1', 'wrong_answer_2')
    @classmethod
    def llm_self_check_answer(cls, v:str, info:ValidationInfo):
        
        print("Verifying...")
        
        content = f"""
            multiple choice question: `{info.data.get('question')}` 
            one of the answers: `{v.answer_text}`
            Given the above multiple choice question, please verify if the given answer is one correct answer to it, or not.
            If you think the answer is {"not" if not v.answer_classification else ""} correct, the answer is now validated, otherwise 
            the answer is not validated. Only if it is not validated, suggest a  new answer which is a negated variant of the original answer.
            """
        
        resp: AnswerVerification = client.chat.completions.create(
            response_model=AnswerVerification,
            messages=[
                {
                    "role": "system",
                    "content": """You are a world class validation model and your job is to validate the answers of a multiple choice test.
                    """,
                },
                {
                    "role": "user",
                    "content": content
                },
            ],
            model="gpt-3.5-turbo",
            temperature=0,
        )  # type: ignore
        print(f"Question: {info.data.get('question')}")
        print(f"Answer: {v.answer_text}")
        print(f"classification: {v.answer_classification}")
        print(f"Verification Response: {resp.model_dump_json(indent=2)}")

        if resp.answer_validated:
            return v
        
        elif resp.maybe_new_answer:
            return Answer(
                answer_text=resp.maybe_new_answer, answer_classification=v.answer_classification
            )
        
        raise ValueError("Answer does not match classification, not able to provide new answer...")

        
topic = "fine tuning of large language models"
number_of_questions = 2
multiple_choice_questions = []

if TEST_MANUALLY:
    multiple_choice_questions.append(
        MultipleChoiceQuestion(
            question="What is fine tuning in the context of large language models?",
            correct_answer=Answer(answer_text="Adapting a pre-trained language model to perform specific tasks by training it on task-specific data.", answer_classification=True),
            wrong_answer_1=Answer(answer_text="Refining the hyperparameters of a language model to improve its performance.", answer_classification=False),
            wrong_answer_2=Answer(answer_text="Breaking down a large language model into smaller components for better efficiency.", answer_classification=False),

        )
    )
else:
    for _ in range(number_of_questions):
        context = [item.question for item in multiple_choice_questions]
        print(context)
        multiple_choice_questions.append( client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=MultipleChoiceQuestion,
            max_retries=2,
            messages=[
                {
                    "role": "system",
                    "content": """You are system that generates interesting and unique multiple choice questions. 
                    The Questions should be on expert level and in the area of the given topic.
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                    Given the topic: `{topic}` \n\nCreate one unique multiple choice question.
                    Remember to only create unique questions with different focus, use the following context to find alredy existing questions.
                    existing questions: ´{context}´
                    """,
                },
            ],
        ))

for item in multiple_choice_questions:
    print(item.model_dump_json(indent=2))

schema = MultipleChoiceQuestion.model_json_schema()


data = []
for question in multiple_choice_questions:
    dct = question.model_dump()
    if hasattr(question,"_raw_response") and question._raw_response.usage is not None:
       usage = question._raw_response.usage.model_dump()
    data.append([val if not isinstance(val, dict) else val.get("answer_text","") for val in dct.values()] + [val for _, val in usage.items()])
    
columns = [key for key, _ in usage.items()]
df = pd.DataFrame(data, columns=['question', 'correct_answer', 'wrong_answer_1', 'wrong_answer_2'] + [key for key, _ in usage.items()])

print(df)

if LOG_WITH_WANDB:
    run.log({"schema": wandb.Table(dataframe=pd.DataFrame([{"schema": schema}]))})

    run.log(
        {
            "usage_total_tokens": df["total_tokens"].sum(),
            "usage_completion_tokens": df["completion_tokens"].sum(),
            "usage_prompt_tokens": df["prompt_tokens"].sum(),
        }
    )

    run.log(
        {
            "results": wandb.Table(dataframe=df),
        }
    )
