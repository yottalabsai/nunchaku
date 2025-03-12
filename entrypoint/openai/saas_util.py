from io import BytesIO
import requests

from entrypoint.openai.protocol import GreenfieldConfig
from log import setup_logging

logger = setup_logging()
async def upload_fileobj_to_greenfield(file: BytesIO, object_name: str, config: GreenfieldConfig):
    try:
        file.seek(0)
        logger.info(f"Uploading file {object_name} to {config.url}:{config.bucket_name}/{object_name}")
        response = requests.post(
            f"{config.url}/api/v1/greenfield/object", 
            data={
                "bucketName": config.bucket_name,
                "storageSource": "greenfield",
                "objectName": object_name
            },
            files={"file": (object_name, file, 'image/png')},
            headers={"x-api-key": config.apikey},
        )
        
        if response.status_code == 200:
            logger.info(response.json())
            logger.info(f"File {object_name} uploaded successfully to {config.bucket_name}/{object_name}")
        else:
            logger.info(response.status_code, response.text)
            logger.error(f"Failed to upload file {object_name}: {response.text}")
    except Exception as e:
        logger.error(f"Error uploading file {object_name}: {e}")
    finally:
        file.close()