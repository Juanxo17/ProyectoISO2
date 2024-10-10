import pydantic
import uuid

from typing_extensions import Optional


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Document(pydantic.BaseModel):
    doc_id: str = pydantic.Field(default_factory=generate_uuid)
    nombre: str
    path: str
    content: str
    user_id: str

class User(pydantic.BaseModel):
    user_id: str = pydantic.Field(default_factory=generate_uuid)
    username: Optional[str]
    password: str
    email: list[str]
    first_name: str
    last_name: Optional[str]
    role: str
