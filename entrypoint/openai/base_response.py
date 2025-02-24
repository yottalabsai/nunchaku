from pydantic import BaseModel


class ImageResponse(BaseModel):
    url: str
    latency: float
    is_safe_prompt: bool
    
class BaseResponse(BaseModel):
    code: int
    message: str
    data: list[ImageResponse]

class ModelStatus(BaseModel):
    status: str
    model: str
    
class HealthCheckResponse(BaseModel):
    code: int
    message: str
    data: ModelStatus

