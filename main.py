import os
import time
from fastapi import FastAPI
from app.routes import user
from app.routes import item
import uvicorn
from fastapi import Depends, FastAPI
from contextlib import asynccontextmanager
from app.config.security import get_current_user, oauth2_scheme
from app.config.database import get_session
from sqlalchemy.orm import Session
from app.routes import tts
from apscheduler.schedulers.background import BackgroundScheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ 
    Load model TTS
    """
    print("Run at startup!")
    classifier = tts.load_model()
    yield
    print("Run on shutdown!")

    
def create_application():
    application = FastAPI(lifespan=lifespan)
    application.include_router(user.user_router)
    application.include_router(user.guest_router)
    application.include_router(user.auth_router)
    application.include_router(item.auth_router)
    application.include_router(tts.tts_router)
    return application

def cleanup_cache():
    """
    Remove files in *basedir* not accessed within *limit* minutes
    :param basedir: directory to clean
    :param limit: minutes
    """
    basedir = os.path.join(os.getcwd(), "cache")
    os.makedirs(basedir, exist_ok=True)
    limit = 6 * 60 * 60
    atime_limit = time.time() - limit
    count = 0
    for filename in os.listdir(basedir):
        path = os.path.join(basedir, filename)
        if os.path.getatime(path) < atime_limit:
            os.remove(path)
            count += 1
    print("Removed {} files.".format(count))

# Schedule cleanup cache with file created >= 6 hours
scheduler = BackgroundScheduler()
# After 30 minutes check file in cache folder if file created >= 6 hours -> remove
scheduler.add_job(cleanup_cache, "interval", minutes=30)
scheduler.start()

app = create_application()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8080)