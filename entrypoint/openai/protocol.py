from pydantic import BaseModel

########################### request ########################
class CreateImageRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 3.5
    lora_name: str = "None"
    lora_weight: float = 0.8
    seed: int = 0
    # sana
    pag_scale: float = 2.0

#### saas request ####
class ObjectCreateRequest(BaseModel):
    bucketName: str    
    storageSource: str
    objectName: str


########################### request ########################


########################### response ########################
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

########################### response ########################


########################### config ########################
class GreenfieldConfig(BaseModel):
    bucket_name: str
    apikey: str
    url: str

class S3Config(BaseModel):
    bucket: str
    prefix_path: str
    aws_access_key_id: str
    aws_secret_access_key: str

class Config(BaseModel):
    greenfield: GreenfieldConfig
    s3: S3Config
########################### config ########################