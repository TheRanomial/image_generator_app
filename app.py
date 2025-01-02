from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import os
import uuid
from urllib.parse import quote

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    prompt: str
    width: int
    height: int 
    enhance: bool
    steps: int 

generated_folder = "generated"
os.makedirs(generated_folder, exist_ok=True)

def generate_image(prompt, model="flux-pro", seed=None, width=1024, height=1024,steps=100,cfg_scale=9,
                   nologo=None, private=None, enhance=None,safe=None):

    """
    Generate a high-quality image using the Pollinations AI API.
    
    Args:
        prompt (str): Main prompt describing the image
        filename (str): Output filename
        width (int): Image width (default: 768)
        height (int): Image height (default: 768)
        model (str): Model to use (default: stable-diffusion-2-1)
        steps (int): Number of denoising steps (default: 50)
        cfg_scale (float): How closely to follow the prompt (default: 7.5)
        enhance (bool): Whether to enhance the image (default: True)
    
    Returns:
        str: The filename of the saved image or None if failed
    """
    
    enhanced_prompt = (
        f"{prompt}, "
        "high resolution, highly detailed, sharp focus, "
        "professional photography, cinematic lighting, "
        "8k uhd, ray tracing, ambient lighting"
    )

    negative_prompt = (
        "blurry, low quality, low resolution, "
        "watermark, signature, oversaturated, "
        "distorted, deformed, pixelated"
    )
    
    unique_filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(generated_folder, unique_filename)
    
    base_url = "https://image.pollinations.ai/prompt"
    encoded_prompt = quote(enhanced_prompt)
    url = f"{base_url}/{encoded_prompt}"

    params = {
        "width": width,
        "height": height,
        "model": model,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler": "DPM++ SDE Karras",  
        "enhance": enhance,
        "nologo": True,
        "negative_prompt": negative_prompt,
        "upscale": True,  
        "upscale_amount": "2",
    }

    try:
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()  
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    if response.status_code == 200:
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ Image successfully saved as {file_path}")
        return unique_filename
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


@app.post("/generate-image")
async def generate_image_endpoint(request: ImageRequest):
   
    filename = generate_image(
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        enhance=request.enhance,
        steps=request.steps,
        nologo=True,
        safe=True,
    )
    if filename:
        return {"file_path": f"/image/{filename}"}
    else:
        raise HTTPException(status_code=500, detail="Image generation failed")

@app.get("/image/{filename}")
async def serve_image(filename: str):
  
    if not filename.endswith('.jpg'):
        filename = f"{filename}.jpg"
    
    file_path = os.path.join(generated_folder, filename)
    abs_file_path = os.path.abspath(file_path)
    
    print(f"Request for file: {filename}")
    print(f"Looking for file at: {abs_file_path}")
    
    if not os.path.exists(abs_file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Image not found. Available files: {os.listdir(generated_folder)}"
        )
    
    try:
        return FileResponse(
            abs_file_path,
            media_type="image/jpeg",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error serving file: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
