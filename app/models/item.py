from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, String, func, ForeignKey
from app.config.database import Base
from sqlalchemy.orm import mapped_column, relationship

class Item(Base):

    __tablename__ = "item"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer,ForeignKey('users.id'))
    name = Column(String(250), nullable=True, index=True, default=None)
    description = Column(String(250), nullable=True, index=True, default=None)

    item = relationship("User")