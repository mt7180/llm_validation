import instructor
from pydantic import (
    BaseModel,
    Field,
    create_model,
)

from dotenv import load_dotenv

from openai import OpenAI
from typing import List, Iterable, Literal


load_dotenv()

client = instructor.patch(OpenAI())


def extract_data_dynamically(context: str, schema_name: str) -> List[BaseModel]:
    class Property(BaseModel):
        """property name and corresponding type"""

        name: str = Field(..., description="must be snake case")
        type: Literal["string", "integer", "boolean"]

    def factory(schema: list[Property]) -> type[BaseModel]:
        type_definition = {
            "string": ("str", ""),
            "integer": ("int", 0),
            "boolean": ("bool", True),
        }
        field_definitions = {
            attribute.name: (type_definition[attribute.type]) for attribute in schema
        }

        print(f"{field_definitions=}")
        return create_model(
            "DataModel",
            **field_definitions,
            __base__=BaseModel,
        )

    custom_properties = client.chat.completions.create(
        # model="gpt-4",
        model="gpt-3.5-turbo",
        response_model=Iterable[Property],
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": "You are a world class data structure extractor.",
            },
            {
                "role": "user",
                "content": f"""provide all necessary properties for a {schema_name} data class which captures all given information
                about the {schema_name}s decribed in the context below. 
                context: {' '.join(context)}
                """,
            },
        ],
    )

    if not custom_properties:
        raise ValueError("schema for dynamic data extraction is empty")

    DataModel: BaseModel = factory(custom_properties)

    def extract(Schema, context) -> List[DataModel]:
        return client.chat.completions.create(
            response_model=Iterable[Schema],
            messages=[
                {"role": "user", "content": f"parse the context: `{context}`"},
            ],
            model="gpt-3.5-turbo",
            temperature=0,
        )  # type: ignore

    return extract(DataModel, context)


if __name__ == "__main__":
    context = """
    Brandon is 33 years old. He works as a solution architect.
    Dominic is 45 years old. He is retired.
    There onces was a prince, named Benny. He ruled for 10 years, which just ended. He started at 22.
    Simon says, why are you 22 years old marvin?
    """

    users = extract_data_dynamically(context, "user")

    for user in users:
        print(user.model_dump_json(indent=2))
