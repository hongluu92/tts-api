from fastapi import FastAPI
from app.routes import user
from app.routes import item
import uvicorn
from fastapi import Depends, FastAPI


def create_application():
    application = FastAPI()
    application.include_router(user.user_router)
    application.include_router(user.guest_router)
    application.include_router(user.auth_router)
    application.include_router(item.auth_router)
    return application


app = create_application()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("test2:app", host='0.0.0.0', port=8080)
