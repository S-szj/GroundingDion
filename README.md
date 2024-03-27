# sam_inapinting.py

（修改图片中旁白的物品，轮廓线条会有所改变）
（可以自定mask区域，在桌布图像上加入物品。）

## 通过指定prompt获取原图中 需要修改的区域
    - image_src - path/to/input.jpg
    - TEXT_PROMPT - 修改物描述
    - out_src - path/to/output.jpg



# sam_tablecloth.py

(修改桌布的纹理样式，保持image_add不变，可以对多路视频不同时间戳做相似改变)

## 配置输入图片路径，设置识别目标的prompt
    - image_src - path/to/input.jpg
    - out_src - path/to/output.jpg

    - image_add - 桌布的纹理，可替换的桌布
    - TEXT_PROMPT - 文本指定数据原图中需要更换的区域（'tablecloth',可以识别多种多样的桌布，不需特别描述）
