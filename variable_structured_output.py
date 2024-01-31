import instructor
from pydantic import (
    BaseModel, 
    Field, 
    ValidationInfo, 
    create_model,
    model_validator,
)

from dotenv import load_dotenv

from openai import OpenAI
from typing import TypeVar, List, Iterable, Optional, Dict, Literal, Type


load_dotenv()

client = instructor.patch(OpenAI())


context = [
    "Brandon is 33 years old. He works as a solution architect.",
    "Jason is 25 years old. He is the GOAT.",
    "Dominic is 45 years old. He is retired.",
    "Jenny is 72. She is a wife and a CEO.",
    "Holly is 22. She is an explorer.",
    "There onces was a prince, named Benny. He ruled for 10 years, which just ended. He started at 22.",
    "Simon says, why are you 22 years old marvin?",
]

def get_model_by_fields(fields):
    return create_model(
        'CustomUser',
        **fields,   #{property_name: property_type]) for property_name, property_type in fields.items()},
        __base__=BaseModel,
    )


CustomUser = TypeVar("CustomUser", bound=BaseModel)


class Property(BaseModel):
    """property name and corresponding type"""
    name: str = Field(...,description="must be snake case")
    type: Literal["string", "integer", "boolean"]


class CustomUserModel(BaseModel):
    """model for user properties"""
    custom_properties: List[Property] = Field(...)
    user_model: Optional[Type[BaseModel]] = None
    users: Optional[List[CustomUser] ] = None

    @model_validator(mode="after")
    def parse_users(self,  info:ValidationInfo):
        print(f"{info=}")
        print(f"{self=}")

        prop_definition = {
            "string": ("str",""),
            "integer": ("int",0),
            "boolean": ("bool", True),
        }
        properties = {prop.name: (prop_definition[prop.type]) for prop in self.custom_properties}
        
        print(f"{properties=}")
        if not properties:
            raise ValueError("user schema empty")
        CustomUser = get_model_by_fields(properties)
       
        self.user_model = CustomUser


        resp = client.chat.completions.create(
            response_model=Iterable[CustomUser],
            messages=[
                
                {
                    "role": "user",
                    "content": f"parse the users: {info}."
                },
            ],
            model="gpt-3.5-turbo",
            temperature=0,
        )  # type: ignore
       
        print(f"inner response: {resp}")
       
        self.users=resp
        return self
    
if __name__ == "__main__":
    from pprint import pprint

    resp = client.chat.completions.create(
        #model="gpt-4",
        model="gpt-3.5-turbo",
        response_model=CustomUserModel,
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": "You are a world class data structure extractor.",
            },
            {
                "role": "user",
                "content": f"""provide all necessary properties for a user data structure which 
                captures all given information about the users decribed in the context below. 
                context: {' '.join(context)}""",
            },
            
        ],
        validation_context={"context": " ".join(context)},
        
    )

    print("\n\nThe Custom User Model:")
    pprint(resp.user_model.model_json_schema())
    print("\n\nThe Parsed Users:")
    for user in resp.users:
        print(user.model_dump_json(indent=2))