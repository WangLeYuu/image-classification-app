---
name: generate-image
description: "Generate AI images using ByteDance Seedream 4.5. Use when you need to: (1) create images from text descriptions, (2) generate project engineering architecture diagrams, (3) generate sequence diagrams, or (4) other scenarios where images may be needed to enhance presentation."
---

# Seedream Image

Generate AI images using ByteDance Seedream 4.5. Use when you need to: (1) create images from text descriptions, (2) generate project engineering architecture diagrams, (3) generate sequence diagrams, or (4) other scenarios where images may be needed to enhance presentation.

## Reference code

Use openai-sdk

```json
import os
from openai import OpenAI


client = OpenAI( 
    base_url="https://ark.cn-beijing.volces.com/api/v3",  
    api_key="08a4779d-df62-4c28-a9ff-081cfc6670c8", 
) 

imagesResponse = client.images.generate( 
    model="ep-20251205103128-7pcc4", 
    prompt="星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车，抢视觉冲击力，电影大片，末日既视感，动感，对比色，oc渲染，光线追踪，动态模糊，景深，超现实主义，深蓝，画面通过细腻的丰富的色彩层次塑造主体与场景，质感真实，暗黑风背景的光影效果营造出氛围，整体兼具艺术幻想感，夸张的广角透视效果，耀光，反射，极致的光影，强引力，吞噬",
    size="2K",
    response_format="url",
    extra_body={
        "watermark": False,
    },
) 

print(imagesResponse.data[0].url)
```

## Request and Response Param
https://www.volcengine.com/docs/82379/1824121?lang=zh


After obtaining the image URL, it needs to be downloaded.