from typing import Any
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from tools.text2image import Text2ImageTool
from dify_plugin import ToolProvider


class DoubaoProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            # Create a simple test prompt for validation
            test_params = {
                "prompt": "测试图像",
                "size": "512x512"
            }
            
            # Try to invoke the tool with test parameters
            for _ in Text2ImageTool.from_credentials(credentials).invoke(
                tool_parameters=test_params
            ):
                pass
        except Exception as e:
            # If any error occurs during validation, raise it as credential validation error
            raise ToolProviderCredentialValidationError(f"Credential validation failed: {str(e)}")
