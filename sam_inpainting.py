import torch
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.models import build_model
from PIL import Image
from segment_anything import build_sam, SamPredictor 
from GroundingDINO.groundingdino.util import box_ops
import numpy as np
from diffusers import StableDiffusionInpaintPipeline

def load_model_local(config_path, model_path, device='cpu'):
    # Load configuration
    args = SLConfig.fromfile(config_path)
    args.device = device
    # Load model state dictionary
    checkpoint = torch.load(model_path, map_location=device)
    # Build model
    model = build_model(args)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(model_path, log))    
    # Set model to evaluation mode
    _ = model.eval()
    return model

def get_masks(BOX_TRESHOLD,TEXT_TRESHOLD,image_src,sam_checkpoint):
    image_source, image = load_image(image_src)

    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    # set image
    sam_predictor.set_image(image_source)

    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
    return masks

if __name__ == '__main__':
    # 加载GroundingDino
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "./ShilongLiu/GroundingDINO/groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "./ShilongLiu/GroundingDINO/GroundingDINO_SwinB.cfg.py"
    sam_checkpoint = './SAM/sam_vit_h_4b8939.pth'
    torch.cuda.set_device(1)
    device = torch.device("cuda")
    groundingdino_model = load_model_local(ckpt_config_filename, ckpt_filenmae,device=device)

    # 通过指定prompt获取原图中 需要修改的区域
    image_src = './test_img/input_img/total/top000125.jpg'
    TEXT_PROMPT = "tablecloth."
    out_src = "./test_img/output_img/result_image.jpg"

    # 获取mask区域
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    masks = get_masks(BOX_TRESHOLD,TEXT_TRESHOLD,image_src,sam_checkpoint)
    
    # 转为PIL格式
    image_mask = Image.fromarray(masks[0][0].cpu().numpy())
    image_original = Image.open(image_src)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "./stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to("cuda")

    # 获取diffusion生成的图片
    image_add = pipe( prompt="high quality.", 
                      negative_prompt="",
                      image=image_original, 
                      mask_image=image_mask,
                      generator = torch.Generator("cuda").manual_seed(1),
                      strength=1,
                      guidance_scale=20,
                      num_inference_steps=100
                    ).images[0]

    # 统一图片尺寸
    image_original = image_original.resize(image_mask.size)
    image_add = image_add.resize(image_mask.size)

    # 创建一个空白图像，尺寸与图像a相同
    result_image = Image.new("RGB", image_mask.size)
    # 将数据原图和创作图片 按mask区域划分 进行拼合
    # 防止原图中关键物体的形状被修改
    for x in range(image_mask.width):
        for y in range(image_mask.height):
            ratio = (255 - image_mask.getpixel((x, y))) / 255.0  
            ori = image_original.getpixel((x, y))
            additive = image_add.getpixel((x, y))
            blended_color = (
                int(ori[0] * ratio + additive[0] * (1 - ratio)),
                int(ori[1] * ratio + additive[1] * (1 - ratio)),
                int(ori[2] * ratio + additive[2] * (1 - ratio))
            )
            result_image.putpixel((x, y), blended_color)

    # 保存拼合后的图像
    result_image.save(out_src)