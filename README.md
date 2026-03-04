**YOLOv11+Deepseek脑瘤系统**（可根据深度学习数据集定制）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0f3c5296d80a4aefa9e19896b5ca51f4.png)

**本系统深度融合YOLO肿瘤目标检测模型+Deepseek大语言模型，实现脑部CT/MRI影像的自动化分析。支持肿瘤区域精准定位、良恶性初步判定、风险等级评估，还具备病例历史记录追溯、诊断报告自动生成、AI问答助手以及病理知识图谱，完全贴合医疗辅助诊断场景需求。**![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/62573724fdba4c499beb1934110884ac.png)


**创新点直击评委老师评分：
多模态AI深度融合：影像检测与医疗语义分析联动，诊断准确率更优
全场景数据接入：支持本地影像文件、医疗设备实时推流、摄像头实时检测
医疗级数据安全：内置影像脱敏、操作日志追溯功能，符合技术规范**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fc5664e0b0454d58a04a726f7f9d6619.png)

**享“落地即用”全套装：
- 完整前后端源码+核心算法模型（附训练数据集与教程文档）
- 零门槛启动包：配套环境安装链接（规避版本冲突）、启动教程、一键运行命令脚本**
- ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/52956b5789454a74b0ad3b6507155254.png)**“YOLOv11+Deepseek 脑瘤智能诊疗系统”**。

，直击当前医疗 AI 领域的痛点：**不仅要“看得见”（检测肿瘤），还要“看得懂”（分析病情、给出建议）。**



---

### 第一部分：项目核心价值与创新点（答辩/文档专用）

在撰写文档或向评委展示时，请重点强调以下三个维度：

#### 1. 多模态深度融合 (Multimodal Fusion)
*   **传统痛点**：传统的医疗 AI 只能画框（检测），无法解释；或者只能聊天（问答），无法看图。
*   **本项目创新**：
    *   **眼（YOLOv11）**：利用 YOLOv11 的高精度和小目标检测能力，在 MRI/CT 影像中毫秒级定位肿瘤区域（如脑膜瘤、胶质瘤），并计算置信度。
    *   **脑（DeepSeek LLM）**：将检测结果（位置、大小、类别）转化为结构化文本，输入给 DeepSeek 大模型。结合内置的医疗知识库，自动生成包含**病情分析、治疗建议、风险预警**的专业报告。
    *   **效果**：实现了从“感知智能”到“认知智能”的跨越。

#### 2. 全场景数据接入能力
*   **灵活性**：系统不仅支持上传本地 DICOM/JPG 影像文件，还支持连接医疗设备推流（RTSP/RTMP）和 USB 摄像头实时演示。
*   **实用性**：模拟了真实的医院工作流——医生可以在阅片室分析历史病例，也可以在手术室通过实时流监控病灶。

#### 3. 动态知识图谱与可解释性
*   **可视化推理**：不仅仅是输出结果，系统还构建了**脑部肿瘤知识图谱**。用户可以点击图谱节点，查看疾病、症状、药物、科室之间的关联关系。
*   **信任机制**：通过 AI 助手引用权威医学文献（如 UpToDate、指南），让每一次诊断建议都有据可依，解决了 AI“黑盒”问题。

---

### 第二部分：系统技术架构解析

清楚如何定制和修改代码，以下是系统的逻辑分层：

| 层级 | 技术栈 | 功能描述 |
| :--- | :--- | :--- |
| **前端交互层** | Vue.js / React + ECharts | 实时视频流播放、检测结果可视化渲染、知识图谱展示、AI 对话界面。 |
| **后端服务层** | Python (FastAPI / Flask) | 接收前端请求，调度 AI 模型，处理业务逻辑，管理用户会话。 |
| **视觉算法层** | **YOLOv11** (Ultralytics) | 加载训练好的 `.pt` 权重，对输入图像进行推理，输出 `[x, y, w, h, class, conf]`。 |
| **语言模型层** | **DeepSeek-V2/V3** (API 或本地部署) | 接收视觉层的 JSON 数据，结合 Prompt 工程，生成自然语言诊断报告。 |
| **数据存储层** | MySQL + Neo4j | MySQL 存储用户信息、历史记录；Neo4j 存储医疗知识图谱关系。 |

---

### 第三部分：核心功能构建代码示例

以下是系统中两个最关键模块的实现逻辑：**YOLOv11 检测集成** 和 **DeepSeek 报告生成**。

#### 1. 视觉检测模块 (YOLOv11 Inference)

此代码展示了如何加载模型并进行推理，同时过滤低置信度结果。

```python
# file: core/vision_detector.py
from ultralytics import YOLO
import cv2
import numpy as np

class BrainTumorDetector:
    def __init__(self, model_path='weights/brain_tumor_yolo11.pt'):
        """
        初始化 YOLOv11 模型
        :param model_path: 训练好的脑瘤检测权重路径
        """
        try:
            self.model = YOLO(model_path)
            print(f"✅ 模型加载成功：{model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            raise e

    def detect(self, image_source):
        """
        执行检测
        :param image_source: 图片路径 或 OpenCV 帧对象 或 视频流
        :return: 检测结果列表 [{'box': [...], 'class': '...', 'conf': 0.95}, ...]
        """
        # 执行推理
        # conf=0.4: 置信度阈值，低于此值视为背景
        # iou=0.45: NMS 阈值，防止重复框选
        results = self.model(image_source, conf=0.4, iou=0.45, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取坐标 [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].cpu().numpy()
                    # 获取类别 ID 和名称
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    # 获取置信度
                    conf = float(box.conf[0])
                    
                    detections.append({
                        'box': xyxy.tolist(),
                        'class': cls_name,
                        'confidence': round(conf, 4)
                    })
        
        return detections

# 使用示例
if __name__ == '__main__':
    detector = BrainTumorDetector()
    # 假设有一张测试图
    res = detector.detect('test_mri_scan.jpg')
    print("🔍 检测结果:", res)
```

#### 2. 智能报告生成模块 (DeepSeek Integration)

此代码展示了如何将 YOLO 的检测结果转化为 Prompt，调用 DeepSeek 生成专业报告。

```python
# file: core/llm_reporter.py
import requests
import json

class MedicalReportGenerator:
    def __init__(self, api_key='YOUR_DEEPSEEK_API_KEY'):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions" # 假设接口地址
        
    def generate_report(self, detection_results, patient_info=None):
        """
        根据检测结果生成医疗报告
        :param detection_results: YOLO 输出的列表
        :param patient_info: 患者基本信息（可选）
        :return: 生成的报告文本
        """
        
        # 1. 构建上下文 Prompt
        # 将检测结果格式化为自然语言描述
        findings_desc = ""
        if not detection_results:
            findings_desc = "未检测到明显的肿瘤区域。"
        else:
            findings_desc = "检测到以下异常区域：\n"
            for i, det in enumerate(detection_results):
                findings_desc += f"{i+1}. 发现 **{det['class']}**，置信度 {det['confidence']*100:.1f}%。\n"
        
        # 2. 构造 System Prompt (角色设定)
        system_prompt = """
        你是一位经验丰富的神经外科专家 AI 助手。
        你的任务是根据影像学检测结果，为患者生成一份专业的初步诊断报告。
        要求：
        1. 语气专业、客观、充满关怀。
        2. 必须包含：影像所见、初步诊断意见、后续检查建议。
        3. 如果未检测到肿瘤，也要建议定期复查。
        4. 严禁给出绝对的确诊结论，必须提示“仅供参考，请以线下医生诊断为准”。
        """
        
        # 3. 构造 User Prompt
        user_prompt = f"""
        【患者信息】: {patient_info if patient_info else '未知'}
        【影像检测结果】: 
        {findings_desc}
        
        请基于以上信息生成诊断报告。
        """
        
        # 4. 调用 DeepSeek API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7, # 控制创造性，医疗场景不宜过高
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            report_content = data['choices'][0]['message']['content']
            return report_content
        except Exception as e:
            return f"⚠️ AI 报告生成失败：{str(e)}"

# 使用示例
if __name__ == '__main__':
    # 模拟 YOLO 的检测结果
    mock_yolo_res = [
        {'box': [100, 100, 200, 200], 'class': 'Meningioma', 'confidence': 0.85},
        {'box': [300, 300, 350, 350], 'class': 'Glioma', 'confidence': 0.62}
    ]
    
    reporter = MedicalReportGenerator()
    report = reporter.generate_report(mock_yolo_res, "男，45 岁，头痛")
    print("📄 生成的报告:\n", report)
```

---

### 第四部分：数据集定制与训练指南

既然系统支持**“根据深度学习数据集定制”**，如果你有自己的医院数据或公开数据集（如 Figshare Brain Tumor Dataset），可以按照以下步骤重新训练 YOLOv11 模型：

1.  **数据准备**：
    *   将 MRI/CT 图像整理为 `images/train`, `images/val`。
    *   使用 LabelImg 或 Roboflow 标注肿瘤区域，保存为 YOLO 格式的 `.txt` 文件。
    *   创建 `data.yaml` 配置文件：
        ```yaml
        path: ./BrainTumorData
        train: images/train
        val: images/val
        nc: 4  # 假设有 4 类：glioma, meningioma, pituitary, no_tumor
        names: ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        ```

2.  **开始训练**：
    ```bash
    yolo detect train data=data.yaml model=yolo11m.pt epochs=100 imgsz=640 batch=16 device=0
    ```
    *   `yolo11m.pt`: 推荐使用 medium 版本，平衡速度和精度。
    *   `epochs=100`: 医疗数据通常较少，需要较多轮次防止欠拟合。

3.  **替换模型**：
    *   训练完成后，将 `runs/detect/train/weights/best.pt` 复制到项目的 `weights/` 目录下，并在 `vision_detector.py` 中修改路径即可。

---

### 第五部分：落地部署建议

为了让系统真正“落地即用”，建议注意以下几点：

1.  **隐私脱敏**：
    *   在上传影像前，务必编写脚本自动裁剪掉图片边缘的患者姓名、ID 等敏感信息（DICOM 标签也需清洗）。
2.  **响应速度优化**：
    *   YOLOv11 推理很快，但 LLM 生成报告可能需要几秒。建议在前端增加“正在分析病情...”的加载动画，或者使用流式输出（Stream）让文字逐字显示，提升用户体验。
3.  **免责声明**：
    *   在所有报告页面显著位置标注：**“本系统结果仅供科研和辅助参考，不能作为最终临床诊断依据。”** 这是医疗 AI 合规的红线。


