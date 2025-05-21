import time
from collections.abc import Generator
import requests
from openai import OpenAI
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin import Tool


class Text2VideoTool(Tool):
    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke text-to-video generation tool using Doubao AI
        """
        # 获取API key
        api_key = self.runtime.credentials.get("api_key")
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        
        # 获取参数
        prompt = tool_parameters.get("prompt", "")
        if not prompt:
            yield self.create_text_message("请输入视频描述提示词")
            return
            
        # 获取比例
        ratio = tool_parameters.get("ratio", "16:9")
        
        # 获取时长
        duration = tool_parameters.get("duration", "5")
        
        # 获取模型
        model = tool_parameters.get("model", "doubao-seedance-1-0-lite-t2v-250428")
        
        # 根据不同模型处理提示词
        if "seedance" in model or "seaweed" in model:
            # 对于豆包模型，添加特殊参数
            # 添加比例参数到提示词
            if ratio and not "--ratio" in prompt:
                prompt = f"{prompt} --ratio {ratio}"
            
            # 添加时长参数到提示词
            if duration and not "--duration" in prompt and not "--dur" in prompt:
                prompt = f"{prompt} --duration {duration}"
        elif "wan2" in model:
            # 对于 Wan2 模型，可能需要不同的处理方式
            # 这里只是示例，可能需要根据实际情况调整
            if "--ratio" not in prompt and "--duration" not in prompt:
                yield self.create_text_message("为 Wan2 模型生成视频，使用默认参数...")
        
        try:
            yield self.create_text_message(f"正在使用{model}模型生成视频...")
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # 第一步：创建视频生成任务
            request_data = {
                "model": model,
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
            
            # 对于不同模型可能需要添加不同的参数
            if "wan2" in model:
                # Wan2 模型需要显式参数而不是在提示词中添加
                parameters = {
                    "duration_seconds": int(duration)
                }
                
                # 使用用户选择的分辨率，如果提供了
                resolution = tool_parameters.get("resolution")
                if resolution:
                    parameters["resolution"] = resolution
                else:
                    # 如果没有提供分辨率，根据比例选择默认值
                    if ratio == "16:9" or ratio == "9:16":
                        parameters["resolution"] = "720p"
                    elif ratio == "4:3" or ratio == "1:1":
                        parameters["resolution"] = "480p"
                    else:
                        parameters["resolution"] = "720p"  # 默认使用 720p
                
                # 使用用户选择的帧率，如果提供了
                fps = tool_parameters.get("fps")
                if fps:
                    parameters["fps"] = int(fps)
                else:
                    # 如果没有提供帧率，使用默认值 16fps
                    parameters["fps"] = 16
                
                # 添加参数到请求数据
                request_data["parameters"] = parameters
            
            response = requests.post(
                f"{base_url}/contents/generations/tasks",
                headers=headers,
                json=request_data
            )
            
            if response.status_code != 200:
                yield self.create_text_message(f"创建视频生成任务失败，状态码: {response.status_code}, 错误信息: {response.text}")
                return
                
            # 获取任务ID
            response_json = response.json()
            task_id = response_json.get("id")
            if not task_id:
                yield self.create_text_message("创建视频生成任务失败，未获取到任务ID")
                return
                
            yield self.create_text_message(f"视频生成任务已创建，任务ID: {task_id}，等待视频生成完成...")
            
            # 第二步：轮询查询任务状态，直到完成或失败
            max_retries = 60  # 最大重试次数，防止无限等待
            retry_count = 0
            video_url = None
            
            while retry_count < max_retries:
                # 等待一段时间再查询
                time.sleep(5)
                
                # 查询任务状态
                task_response = requests.get(
                    f"{base_url}/contents/generations/tasks/{task_id}",
                    headers=headers
                )
                
                if task_response.status_code != 200:
                    yield self.create_text_message(f"查询视频生成任务失败，状态码: {task_response.status_code}, 错误信息: {task_response.text}")
                    return
                    
                task_data = task_response.json()
                
                # 检查任务状态
                status = task_data.get("status")
                
                if status == "succeeded":
                    # 任务成功，获取视频URL
                    video_url = task_data.get("content", {}).get("video_url")
                    break
                elif status == "failed":
                    # 任务失败
                    error_message = task_data.get("error", {}).get("message", "未知错误")
                    yield self.create_text_message(f"视频生成任务失败: {error_message}")
                    return
                elif status == "canceled":
                    # 任务被取消
                    yield self.create_text_message("视频生成任务已被取消")
                    return
                
                # 继续等待
                yield self.create_text_message(f"视频正在生成中，已等待 {(retry_count + 1) * 5} 秒...")
                retry_count += 1
            
            # 检查是否获取到视频URL
            if video_url:
                yield self.create_text_message("视频生成成功！")
                yield self.create_text_message(f"视频链接: {video_url}")
                
                # 创建带有视频链接的消息
                video_data = {
                    "type": "video",
                    "url": video_url
                }
                yield self.create_json_message(video_data)
            else:
                yield self.create_text_message("视频生成超时或失败，请稍后再试")
        
        except Exception as e:
            # 处理异常
            yield self.create_text_message(f"生成视频时出错: {str(e)}")
