import torch,os
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.models import build_model
from PIL import Image
from segment_anything import build_sam, SamPredictor 
from GroundingDINO.groundingdino.util import box_ops
import numpy as np

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

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "./ShilongLiu/GroundingDINO/groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "./ShilongLiu/GroundingDINO/GroundingDINO_SwinB.cfg.py"
sam_checkpoint = './SAM/sam_vit_h_4b8939.pth'

torch.cuda.set_device(1)
device = torch.device("cuda")
groundingdino_model = load_model_local(ckpt_config_filename, ckpt_filenmae,device=device)


# images_src = '/home/shizijie/experiment/Grounded-Segment-Anything-main/test_img/input_img/example/faceImg/'
images_src = '/home/shizijie/experiment/Grounded-Segment-Anything-main/test_img/input_img/example/leftImg/'
# images_src = '/home/shizijie/experiment/Grounded-Segment-Anything-main/test_img/input_img/example/rightImg/'
# images_src = '/home/shizijie/experiment/Grounded-Segment-Anything-main/test_img/input_img/example/topImg/'
# images_src = '/home/shizijie/experiment/Grounded-Segment-Anything-main/test_img/input_img/example/wallImg/'


names = [f for f in os.listdir(images_src) if f.lower().endswith('.jpg')]
for name in names:
    print(name)
    image_src = images_src+name 

    TEXT_PROMPT = "tablecloth."
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    masks = get_masks(BOX_TRESHOLD,TEXT_TRESHOLD,image_src,sam_checkpoint)

    image_mask = masks[0][0].cpu().numpy()
    image_mask_pil = Image.fromarray(image_mask)
    image_a = image_mask_pil
    # image_a = Image.open('/home/shizijie/experiment/Grounded-Segment-Anything-main/sam_img/all.jpg')
    image_b = Image.open(image_src)
    image_c = Image.open("./test_img/input_img/table_cloth/Stainless steel material, smooth texture, desktop.jpg")
    image_b_resized = image_b.resize(image_a.size)
    image_c_resized = image_c.resize(image_a.size)

    # 创建一个空白图像，尺寸与图像a相同
    result_image = Image.new("RGB", image_a.size)
    # 将图像b按照图像a中的黑色部分进行拼合
    for x in range(image_a.width):
        for y in range(image_a.height):
            ratio = (255 - image_a.getpixel((x, y))) / 255.0  
            color_b = image_b_resized.getpixel((x, y))
            color_c = image_c_resized.getpixel((x, y))
            blended_color = (
                int(color_b[0] * ratio + color_c[0] * (1 - ratio)),
                int(color_b[1] * ratio + color_c[1] * (1 - ratio)),
                int(color_b[2] * ratio + color_c[2] * (1 - ratio))
            )
            result_image.putpixel((x, y), blended_color)

    # 保存拼合后的图像
    # result_image.save("./test_img/output_img/example/faceImg/"+name)
    result_image.save("./test_img/output_img/leftImg/"+name)
    # result_image.save("./test_img/output_img/example/rightImg/"+name)
    # result_image.save("./test_img/output_img/example/topImg/"+name)
    # result_image.save("./test_img/output_img/example/wallImg/"+name)