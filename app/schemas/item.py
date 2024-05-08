from pydantic import BaseModel

class ItemResponce(BaseModel):
    name:str
    description:str
    access_token:str
    refresh_token:str
    expires_in: int

class RegisterItemRequest(BaseModel):
    name: str
    description: str

class UpdateItemRequest(BaseModel):
    old_name:str
    new_name:str
    description: str
