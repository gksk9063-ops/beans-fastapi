from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass


def init_db():
    from . import models  # noqa: F401  -- 모델 등록
    Base.metadata.create_all(bind=engine)
    # create_all은 이미 있는 테이블에 컬럼을 추가하지 않는다 → 기존 PostgreSQL 테이블에 새 컬럼 2개 추가 (이력 행은 NULL로 보존)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE image_jobs ADD COLUMN IF NOT EXISTS leaf_distance DOUBLE PRECISION"))
            conn.execute(text("ALTER TABLE image_jobs ADD COLUMN IF NOT EXISTS maybe_not_bean BOOLEAN"))
