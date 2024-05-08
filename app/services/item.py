from datetime import datetime, timedelta
from sqlalchemy.orm import joinedload
from sqlalchemy import and_
from fastapi import HTTPException
from app.models.item import Item
from app.config.settings import get_settings
from app.config.security import generate_token, get_token_payload, hash_password, is_password_strong_enough, load_user, str_decode, str_encode, verify_otp, verify_password
from app.models.user import User, UserToken
from app.utils.email_context import FORGOT_PASSWORD, USER_VERIFY_ACCOUNT
from app.utils.string import unique_string


settings = get_settings()


async def get_refresh_token(refresh_token, session):
    token_payload = get_token_payload(refresh_token, settings.SECRET_KEY, settings.JWT_ALGORITHM)
    if not token_payload:
        raise HTTPException(status_code=400, detail="Invalid Request.")
    
    refresh_key = token_payload.get('t')
    access_key = token_payload.get('a')
    user_id = str_decode(token_payload.get('sub'))
    user_token = session.query(UserToken).options(joinedload(UserToken.user)).filter(UserToken.refresh_key == refresh_key,
                                                 UserToken.access_key == access_key,
                                                 UserToken.user_id == user_id,
                                                 UserToken.expires_at > datetime.now()
                                                 ).first()
    if not user_token:
        raise HTTPException(status_code=400, detail="Invalid Request.")
    
    user_token.expires_at = datetime.now()
    session.add(user_token)
    session.commit()
    return _generate_tokens(user_token.user, session)


def _generate_tokens(user, session):
    refresh_key = unique_string(100)
    access_key = unique_string(50)
    rt_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)

    user_token = UserToken()
    user_token.user_id = user.id
    user_token.refresh_key = refresh_key
    user_token.access_key = access_key
    user_token.expires_at = datetime.now() + rt_expires
    session.add(user_token)
    session.commit()
    session.refresh(user_token)

    at_payload = {
        "sub": str_encode(str(user.id)),
        'a': access_key,
        'r': str_encode(str(user_token.id)),
        'n': str_encode(f"{user.name}")
    }

    at_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = generate_token(at_payload, settings.JWT_SECRET, settings.JWT_ALGORITHM, at_expires)

    rt_payload = {"sub": str_encode(str(user.id)), "t": refresh_key, 'a': access_key}
    refresh_token = generate_token(rt_payload, settings.SECRET_KEY, settings.JWT_ALGORITHM, rt_expires)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": at_expires.seconds
    }

async def create_item(data,user,session):
    item_exist = session.query(Item).filter(Item.name == data.name).first()
    if item_exist:
        raise HTTPException(status_code=400, detail="App is already exists.")
    item = Item()
    item.name = data.name
    item.description = data.description
    item.user_id = user.id
    session.add(item)
    session.commit()
    session.refresh(item)
    token = _generate_tokens(user, session)
    token.update({'name':data.name})
    token.update({'description':data.description})
    return token

async def update_item(user, data, session):
    item = session.query(Item).filter(and_(Item.name == data.old_name,Item.user_id==user.id)).first()
    if item is None:
        raise HTTPException(status_code=404, detail="App not found.")
    item.name = data.new_name
    item.description = data.description
    session.commit()
    session.refresh(item)
    return {"message": "App updated successfully."}


async def delete_item(item_name,user, session):
    item = session.query(Item).filter(and_(Item.name == item_name,Item.user_id==user.id)).first()
    if item is None:
        raise HTTPException(status_code=404, detail="App not found.")
    session.delete(item)
    session.commit()
    return {"message": "App deleted successfully."}

async def get_items(user,session):
    items = session.query(Item).filter(Item.user_id==user.id).all()
    return items

async def get_item(item_name,user,session):
    item = session.query(Item).filter(and_(Item.name == item_name,Item.user_id==user.id)).first()
    return item