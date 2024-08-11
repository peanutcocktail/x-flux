import gradio as gr
from PIL import Image
import os
from src.flux.xflux_pipeline import XFluxPipeline
def run(prompt):
    if args.image:
        image = Image.open(args.image)
    else:
        image = None

    model_type = "flux-dev-fp8"
    device = "mps"
    offload = True
    lora_repo_id= "XLabs-AI/flux-lora-collection"
    lora_name="realism_lora.safetensors"
    seed = 123456789
    use_lora = True
    lora_weight = 0.9
    use_controlnet = False
    image = None
    width = 512
    height = 512
    guidance = 3.5
    steps = 50

    xflux_pipeline = XFluxPipeline(
      model_type,
      device,
      offload,
      seed
    )
    if use_lora:
        print('load lora:', args.lora_repo_id, args.lora_name)
        xflux_pipeline.set_lora(
          None,
          lora_repo_id,
          lora_name,
          lora_weight
        )
    elif use_controlnet:
        xflux_pipeline.set_controlnet(
          "canny",
          local_path,
          repo_id, 
          name
        )

    result = xflux_pipeline(
      prompt,
      controlnet_image=image,
      width=width,
      height=height,
      guidance=guidance,
      num_steps=steps,
    )
    return result
#    if not os.path.exists(save_path):
#        os.mkdir(save_path)
#    ind = len(os.listdir(args.save_path))
#    result.save(os.path.join(args.save_path, f"result_{ind}.png"))


demo = gr.Interface(
    fn=run,
    inputs=["text",],
    outputs=["image"],
)

demo.launch()
