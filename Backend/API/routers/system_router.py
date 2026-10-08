from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from ..schemas import SystemStatusOut
from ..services import system_check

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status", response_model=SystemStatusOut)
async def get_status():
    # Threadpool: the first call probes the GPU in a subprocess and imports
    # every stage module for its model paths - seconds, not milliseconds.
    return await run_in_threadpool(system_check.status)
