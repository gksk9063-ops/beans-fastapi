"""비동기 작업 목록 조회 (제공 코드에 추가한 기능). 기존 router_async.py는 수정하지 않는다."""
from fastapi import APIRouter, Query, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from .config import BASE_DIR
from .database import SessionLocal
from .models import ImageJob

router = APIRouter(tags=["작업 목록"])
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/api/async/jobs/")
def list_jobs(limit: int = Query(20, ge=1, le=100)):
    """최근 비동기 작업을 최신순으로 돌려준다."""
    with SessionLocal() as db:
        jobs = db.scalars(select(ImageJob).order_by(ImageJob.id.desc()).limit(limit)).all()
        return [
            {
                "job_id": j.id,
                "status": j.status,
                "model_id": j.model_id,
                "top1_label": j.top1_label,
                "top1_score": j.top1_score,
                "inference_ms": j.inference_ms,
                "cache_hit": j.cache_hit,
                "error_message": j.error_message,
                "maybe_not_bean": j.maybe_not_bean,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in jobs
        ]


@router.get("/vision/jobs/")
def jobs_page(request: Request):
    return templates.TemplateResponse(request=request, name="jobs.html")
