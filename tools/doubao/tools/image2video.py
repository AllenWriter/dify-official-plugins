import time
from collections.abc import Generator
import requests
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin import Tool


class Image2VideoTool(Tool):
    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke image-to-video generation tool using Doubao AI
        
        Parameters:
            tool_parameters (dict): Dictionary containing:
                - prompt (str): Text description for video generation
                - image_url (str): URL of the image to be used for video generation
                - duration (str): Video duration in seconds
                - ratio (str): Aspect ratio (e.g., "16:9")
        
        Returns:
            Generator[ToolInvokeMessage, None, None]: Messages including video generation progress and final video URL
        """
        # 获取API key
        api_key = self.runtime.credentials.get("api_key")
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        
        # 获取参数
        prompt = tool_parameters.get("prompt", "")
        if not prompt:
            yield self.create_text_message("请输入视频描述提示词")
            return
            
        # 获取图片URL
        image_url = tool_parameters.get("image_url", "")
        if not image_url:
            yield self.create_text_message("请提供图片URL")
            return
        
        # 获取比例
        ratio = tool_parameters.get("ratio", "16:9")
        # 添加比例参数到提示词
        if ratio and not "--ratio" in prompt:
            prompt = f"{prompt} --ratio {ratio}"
        
        # 获取时长
        duration = tool_parameters.get("duration", "5")
        # 添加时长参数到提示词
        if duration and not "--duration" in prompt and not "--dur" in prompt:
            prompt = f"{prompt} --duration {duration}"
        
        # 使用图生视频模型
        model = "doubao-seedance-1-0-lite-i2v-250428"
        
        try:
            # 显示正在使用的模型
            yield self.create_text_message("正在使用豆包 Seedance 图生视频模型生成视频...")
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # 创建请求内容
            content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
            
            # 请求数据
            request_data = {
                "model": model,
                "content": content
            }
            
            # 发送请求
            yield self.create_text_message("正在创建视频生成任务...")
            
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
                
            # 显示任务信息
            yield self.create_text_message(f"视频生成任务已创建，任务ID: {task_id}")
            yield self.create_text_message(f"提示词: {prompt}")
            yield self.create_text_message(f"输入图片: {image_url}")
            yield self.create_text_message("正在等待视频生成完成...")
            
            # 轮询查询任务状态，直到完成或失败
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
