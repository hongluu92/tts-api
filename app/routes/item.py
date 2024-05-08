
from fastapi import APIRouter, BackgroundTasks, Depends, status, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config.database import get_session
from app.schemas.item import RegisterItemRequest,ItemResponce,UpdateItemRequest
from app.services import item
from app.config.security import get_current_user, oauth2_scheme

auth_router = APIRouter(
    prefix="/apps",
    tags=["Apps"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(oauth2_scheme), Depends(get_current_user)]
)

@auth_router.post("/create-item", status_code=status.HTTP_200_OK,response_model=ItemResponce)
async def create_item(data: RegisterItemRequest,user = Depends(get_current_user),session: Session = Depends(get_session)):
    return await item.create_item(data, user, session)

@auth_router.post("/update-item", status_code=status.HTTP_200_OK)
async def update_item(data: UpdateItemRequest,user = Depends(get_current_user),session: Session = Depends(get_session)):
    return await item.update_item(user, data, session)

@auth_router.get("/delete-item", status_code=status.HTTP_200_OK)
async def delete_item(item_name,session: Session = Depends(get_session),user = Depends(get_current_user)):
    return await item.delete_item(item_name, user, session)

@auth_router.get("/item", status_code=status.HTTP_200_OK)
async def get_item(item_name,session: Session = Depends(get_session),user = Depends(get_current_user)):
    return await item.get_item(item_name, user, session)

@auth_router.get("/items", status_code=status.HTTP_200_OK)
async def get_all_item(session: Session = Depends(get_session),user = Depends(get_current_user)):
    return await item.get_items(user, session)


