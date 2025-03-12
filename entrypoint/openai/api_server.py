import asyncio
from io import BytesIO
import json
import os
import resource
import signal
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from contextlib import asynccontextmanager
from http import HTTPStatus
import time
from typing import Any, Optional
import uuid

import psutil
import uvicorn
import uvloop
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import torch
from protocol import (
    BaseResponse,
    CreateImageRequest,
    HealthCheckResponse,
    ImageResponse,
    ModelStatus,
    Config,
    S3Config,
    GreenfieldConfig,
)
from entrypoint import load_pipeline
from entrypoint.openai.log import setup_logging
from entrypoint.vars import PROMPT_TEMPLATES, MODEL_MAPPINGS
from nunchaku.models.safety_checker import SafetyChecker
import s3_util  
import saas_util
from dotenv import load_dotenv

VERSION = "1.0.0"
TIMEOUT_KEEP_ALIVE = 180  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = setup_logging()

router = APIRouter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("start server")
        yield
        
    finally:
        # Ensure app state including engine ref is gc'd
        torch.cuda.empty_cache()
        logger.info("stop server empty cuda")
        del app.state

@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    state = raw_request.app.state
    model_status = ModelStatus(model=state.model_name, status="ok")
    result = HealthCheckResponse(code=10000, message="success", data=model_status)
    logger.info(f"Health check response {result.model_dump()}")
    return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.OK)

@router.get("/version")
async def show_version():
    version = {"version": VERSION}
    return JSONResponse(content=version)

@router.api_route("/v1/images/generations", methods=["GET", "POST"])
async def imagesGenerations(req: CreateImageRequest, raw_req: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    state = raw_req.app.state
    prompt = req.prompt
    is_safe_prompt = True
    logger.info(f"req: {req}")
    try:
        if not state.safety_checker(prompt):
            prompt = "A peaceful world."
            is_safe_prompt = False
            logger.info("Unsafe prompt detected")
        start_time = time.time()
        image = generate_image(req=req, raw_req=raw_req, prompt=prompt)
        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"start_time: {start_time}, end_time: {end_time}, latency: {latency}")
        
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        greenfield_bytes = BytesIO()
        image.save(greenfield_bytes, format="PNG")
        greenfield_bytes.seek(0)
    except Exception as e:
        logger.exception("imagesGenerations failed")
        result = BaseResponse(code=10001, message="failed to generation image", data=[])    
        return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    finally:
        del image
        torch.cuda.empty_cache()

    s3Config: S3Config = state.config.s3
    object_name = s3Config.prefix_path + f"{state.model}-{state.precision}-{uuid.uuid4()}.png"
    s3_client = state.s3_client

    url = s3_util.upload_file_and_get_presigned_url(s3_client, s3Config.bucket, object_name, image_bytes)
    if url is not None:
        image_response = ImageResponse(url=url, latency=latency, is_safe_prompt=is_safe_prompt)
        result = BaseResponse(code=10000, message="success", data=[image_response])
        asyncio.create_task(saas_util.upload_fileobj_to_greenfield(greenfield_bytes, object_name, state.config.greenfield))
    else:
        result = BaseResponse(code=10001, message="failed to generation image", data=[])    
    return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.OK)

def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        logger.error(exc)
        return JSONResponse(content="BAD_REQUEST", status_code=HTTPStatus.BAD_REQUEST)

    return app

async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("nunchaku API server version %s", VERSION)
    logger.info("args: %s", args)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)
    app = build_app(args)
    pipeline = load_pipeline.get_pipeline(args.model, args.precision, args.use_qencoder, args.lora_name, args.lora_weight)
    logger.info("Loaded pipeline")
    init_app_state(app.state, pipeline, args)
    logger.info("Initialized app state")
    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        **uvicorn_kwargs,
    )
    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ', '.join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()

def find_process_using_port(port: int) -> Optional[psutil.Process]:
    # TODO: We can not check for running processes with network
    # port on macOS. Therefore, we can not have a full graceful shutdown
    # of vLLM. For now, let's not look for processes in this case.
    # Ref: https://www.florianreinhard.de/accessdenied-in-psutil/
    if sys.platform.startswith("darwin"):
        return None

    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:
    """Adds handlers for fatal errors that should crash the server"""

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, __):
        logger.fatal("RuntimeError, terminating server "
                         "process")
        server.should_exit = True
        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630 # noqa: E501
def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type,
                               (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase"
                "with error %s. This can cause fd limit errors like"
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n", current_soft, e)

def generate_image(req: CreateImageRequest, raw_req: Request, prompt: str):
    state = raw_req.app.state
    model = state.model
    pipeline = state.pipeline
    height = req.height if req.height != 0 else 1024
    width = req.width if req.width != 0 else 1024
    pag_scale = req.pag_scale if req.pag_scale != 0 else 2.0
    if model in ["schnell", "dev"]:
        lora_name = state.lora_name
        prompt = PROMPT_TEMPLATES[lora_name].format(prompt=prompt)
        logger.info(f"generate_image: model={model}, prompt={prompt}, height={height}, width={width}, num_inference_steps={req.num_inference_steps}, guidance_scale={req.guidance_scale} seed={req.seed}")
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=torch.Generator().manual_seed(req.seed),
        ).images[0]
    elif model in ["sana"]:
        logger.info(f"generate_image: model={model}, prompt={prompt}, height={height}, width={width}, guidance_scale={req.guidance_scale}, pag_scale={pag_scale}, num_inference_steps={req.num_inference_steps}, seed={req.seed}")
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=req.guidance_scale,
            pag_scale=pag_scale,
            num_inference_steps=req.num_inference_steps,
            generator=torch.Generator().manual_seed(req.seed),
        ).images[0]
    return image

def read_config():
    load_dotenv()
    bucket = os.getenv("S3_BUCKET")
    prefix_path = os.getenv("S3_PREFIX_PATH")
    aws_access_key_id = os.getenv("S3_AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
    greenfield_bucket = os.getenv("SAAS_GREENFIELD_BUCKET_NAME")
    greenfield_api_key = os.getenv("SAAS_GREENFIELD_APIKEY")
    greenfield_url = os.getenv("SAAS_GREENFIELD_URL")

    # Non-empty checks for environment variables
    if not bucket:
        raise ValueError("S3_BUCKET environment variable must be set and not empty.")
    if not prefix_path:
        raise ValueError("S3_PREFIX_PATH environment variable must be set and not empty.")
    if not aws_access_key_id:
        raise ValueError("S3_AWS_ACCESS_KEY_ID environment variable must be set and not empty.")
    if not aws_secret_access_key:
        raise ValueError("S3_AWS_SECRET_ACCESS_KEY environment variable must be set and not empty.")
    if not greenfield_bucket:
        raise ValueError("SAAS_GREENFIELD_BUCKET_NAME environment variable must be set and not empty.")
    if not greenfield_api_key:
        raise ValueError("SAAS_GREENFIELD_APIKEY environment variable must be set and not empty.")
    if not greenfield_url:
        raise ValueError("SAAS_GREENFIELD_URL environment variable must be set and not empty.")
    
    s3 = S3Config(
        bucket=bucket,
        prefix_path=prefix_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    greenfield = GreenfieldConfig(
        bucket_name=greenfield_bucket,
        url=greenfield_url,
        apikey=greenfield_api_key,
    )
    config = Config(s3=s3, greenfield=greenfield)
    return config

def init_app_state(app_state, pipeline, args):
    app_state.model = args.model
    app_state.precision = args.precision
    app_state.model_name = MODEL_MAPPINGS[app_state.model][app_state.precision]
    app_state.pipeline = pipeline
    app_state.lora_name = args.lora_name
    logger.info("load config")
    app_state.config = read_config()
    logger.info("get config done")
    app_state.s3_client = s3_util.get_s3_client(app_state.config.s3)
    logger.info(f"start init safety checker {args.no_safety_checker}")
    app_state.safety_checker = SafetyChecker("cuda", disabled=args.no_safety_checker)
    logger.info("end init safety checker")

def mark_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "-m", "--model", type=str, default="schnell", choices=["schnell", "dev", "sana"], help="Which model to use"
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="int4",
        choices=["int4", "bf16"],
        help="Which precisions to use",
    )
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder", default=False)
    parser.add_argument("--lora-name", default="None", choices=["None", "All", "Anime", "GHIBSKY Illustration", "Realism", "Yarn Art", "Children Sketch"])
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument("--no-safety-checker", action="store_true", help="Disable safety checker", default=False)

    parser.add_argument("--allowed-origins", type=list, default=["*"])
    parser.add_argument("--allow-credentials", type=bool, default=True)
    parser.add_argument("--allowed-methods", type=list, default=["*"])
    parser.add_argument("--allowed-headers", type=list, default=["*"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    mark_args(parser)
    args = parser.parse_args()

    uvloop.run(run_server(args))
