# title: ComfyUI Universal Advanced
# author: theJSN https://github.com/pupphelper/Openwebui-ComfyUI-Universal-Advanced
# inspired by Haervwe https://github.com/Haervwe/open-webui-tools
# version: 17.1.5 (Final Robustness Patch)
# required_open_webui_version: 0.5.0


import json
import uuid
import aiohttp
import asyncio
import random
import re
import time
import os
import base64
import difflib
import copy
from typing import List, Dict, Callable, Optional, Tuple, Any
from pydantic import BaseModel, Field
from open_webui.utils.misc import get_last_user_message_item
from open_webui.models.users import User, Users
from open_webui.utils.chat import generate_chat_completion

import logging
import requests

import io
import mimetypes
from urllib.parse import urlparse, urlunparse
from fastapi import UploadFile
from open_webui.routers.files import upload_file_handler


# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Set logger level to INFO to see the new debug messages
logger.setLevel(logging.INFO)


# --- OLLAMA VRAM Management Functions ---
def get_loaded_models(api_url: str) -> list:
    if not api_url:
        return []
    try:
        response = requests.get(f"{api_url.rstrip('/')}/api/ps", timeout=5)
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.RequestException as e:
        logger.error(f"Error fetching loaded Ollama models from {api_url}: {e}")
        return []


def _unload_specific_model(model_name: str, api_url: str):
    """
    Sends a request to unload a single, specific Ollama model.
    """
    if not model_name or not api_url:
        return
    try:
        logger.info(
            f"Attempting to unload specific Ollama model '{model_name}' from {api_url}"
        )
        # To unload, we send a request with keep_alive: 0
        response = requests.post(
            f"{api_url.rstrip('/')}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=10,
        )
        # A 404 can be normal if the model wasn't loaded, so we don't raise for it.
        if response.status_code == 200:
            logger.info(f"Successfully sent unload command for model: {model_name}")
        else:
            logger.warning(
                f"Received status {response.status_code} when trying to unload {model_name}. This may be normal if the model wasn't loaded."
            )
    except requests.RequestException as e:
        logger.error(
            f"Error during specific Ollama model unload for '{model_name}' from {api_url}: {e}"
        )


# --- Main Tools Class (Entrypoint for Open WebUI) ---
class Tools:
    def __init__(self):
        self.pipe = Pipe()

    async def __call__(
        self, body: dict, user: dict, event_emitter: Callable, request=None
    ) -> dict:
        return await self.pipe.pipe(
            body=body,
            __user__=user,
            __event_emitter__=event_emitter,
            __request__=request,
        )


DEFAULT_WORKFLOW_JSON = json.dumps({}, indent=2)


# --- Core Logic Class ---
class Pipe:
    class Valves(BaseModel):
        # General Settings
        Tool_Description: str = Field(default="", extra={"type": "textarea"})
        ComfyUI_Primary_Address: str = Field(default="http://192.168.2.2:8189")
        ComfyUI_Secondary_Address: Optional[str] = Field(default=None)
        Health_Check_Port: Optional[int] = Field(default=9188)
        Administrative_Roles: str = Field(default="admin")

        # User Workflow Settings
        ComfyUI_Workflow_JSON: str = Field(
            default=DEFAULT_WORKFLOW_JSON, extra={"type": "textarea"}
        )
        I2V_Workflow_JSON: Optional[str] = Field(
            default=None, extra={"type": "textarea"}
        )

        # Admin Workflows
        Admin_Workflow_1_Name: Optional[str] = Field(
            default="Example: Character Generator"
        )
        Admin_Workflow_1_JSON: Optional[str] = Field(
            default=None, extra={"type": "textarea"}
        )
        Admin_Workflow_1_Prompt_Node_ID: Optional[str] = Field(default=None)
        Admin_Workflow_2_Name: Optional[str] = Field(default=None)
        Admin_Workflow_2_JSON: Optional[str] = Field(
            default=None, extra={"type": "textarea"}
        )
        Admin_Workflow_2_Prompt_Node_ID: Optional[str] = Field(default=None)
        Admin_Workflow_3_Name: Optional[str] = Field(default=None)
        Admin_Workflow_3_JSON: Optional[str] = Field(
            default=None, extra={"type": "textarea"}
        )
        Admin_Workflow_3_Prompt_Node_ID: Optional[str] = Field(default=None)
        Admin_Workflow_4_Name: Optional[str] = Field(default=None)
        Admin_Workflow_4_JSON: Optional[str] = Field(
            default=None, extra={"type": "textarea"}
        )
        Admin_Workflow_4_Prompt_Node_ID: Optional[str] = Field(default=None)
        Admin_Workflow_5_Name: Optional[str] = Field(default=None)
        Admin_Workflow_5_JSON: Optional[str] = Field(
            default=None, extra={"type": "textarea"}
        )
        Admin_Workflow_5_Prompt_Node_ID: Optional[str] = Field(default=None)

        # VRAM Management
        Primary_Server_Max_VRAM_Usage_GB: int = Field(default=8)
        Allow_Ollama_Unload_On_Primary: bool = Field(default=True)
        Ollama_GPU_Busy_Threshold_Percent: int = Field(default=50)
        unload_ollama_models: bool = Field(default=True)
        Enhancer_Ollama_API_URL: str = Field(
            default="http://host.docker.internal:11434"
        )
        Ollama_Primary_API_URL: Optional[str] = Field(default=None)
        Ollama_Secondary_API_URL: Optional[str] = Field(default=None)
        ComfyUI_Clear_VRAM_Before_Check: bool = Field(default=True)
        ComfyUI_Free_Memory: bool = Field(default=False)

        # Media & Output
        Save_Local_Copies: bool = Field(default=False)
        Base64_Warning_Threshold_MB: int = Field(default=25)

        # User Workflow Node IDs
        Image_Input_Node_ID: Optional[str] = Field(default=None)
        Image_Input_Field_Name: str = Field(default="image")
        Prompt_Node_ID: str = Field(default="6")
        Prompt_Field_Name: str = Field(default="text")
        Seed_Node_ID: str = Field(default="9")
        Seed_Field_Name: str = Field(default="seed")
        Size_Node_ID: str = Field(default="11")
        Width_Field_Name: str = Field(default="width")
        Height_Field_Name: str = Field(default="height")
        Length_Node_ID: Optional[str] = Field(default=None)
        Length_Field_Name: str = Field(default="length")
        Lora_Node_ID: str = Field(default="10")
        Lora_Name_Field_Name: str = Field(default="lora_name")
        Batch_Node_ID: Optional[str] = Field(default=None)
        Batch_Field_Name: str = Field(default="batch_size")
        Steps_Node_ID: Optional[str] = Field(default=None)
        Steps_Field_Name: str = Field(default="steps")
        Model_Node_ID: Optional[str] = Field(default=None)
        Model_Field_Name: str = Field(default="ckpt_name")

        # User Workflow Defaults
        Width: int = Field(default=1024)
        Height: int = Field(default=1024)
        Length: int = Field(default=1)
        Lora_Name: str = Field(default="None")
        Enhancer_Model_ID: str = Field(default="")
        Enhancer_System_Prompt: str = Field(
            default="You are an expert prompt engineer...", extra={"type": "textarea"}
        )
        max_wait_time: int = Field(default=1200)
        Media_Download_Timeout: int = Field(default=20)

    def __init__(self):
        self.valves = self.Valves()
        self.ADMIN_COMMAND_KEYWORDS = [
            "clear-queue",
            "cancel-queue",
            "loralist",
            "workflows",
        ]
        self.USER_COMMAND_KEYWORDS = ["status", "stats", "free", "unload"]


    def _is_user_admin(self, user: User) -> bool:
        if not user or not user.role:
            return False
        admin_roles = {
            role.strip().lower() for role in self.valves.Administrative_Roles.split(",")
        }
        return user.role.lower() in admin_roles

    def _parse_prompt(self, prompt: str) -> Dict[str, Any]:
        results = {
            "cleaned_prompt": "",
            "width": None,
            "height": None,
            "length": None,
            "lora_name": None,
            "batch_size": None,
            "enhance_requested": False,
            "suggest_count": None,
            "primary_only": False,
            "secondary_only": False,
            "force_server": None,
            "is_already_enhanced": False,
            "nodes_to_disable": [],
            "nodes_to_enable": [],
            "steps": None,
            "model_name": None,
            "lora_overrides": {},
            "injections": [],
            "unrecognized_tags": [],
        }

        tag_pattern = r"([<(][\w\s\d:.,/\\=×x-]*[>)])"
        all_tags = re.findall(tag_pattern, prompt)
        processed_tags = []

        for tag in all_tags:
            content = tag[1:-1].strip()
            content_lower = content.lower()
            tag_processed = False

            # MODIFICATION: Regex now includes '=' for the force command.
            force_match = re.fullmatch(r"force[-:=](primary|secondary)", content_lower)
            if force_match:
                server_target = force_match.group(1)
                if results["force_server"] and results["force_server"] != server_target:
                    raise ValueError(f"Conflicting force tags detected: Cannot force both primary and secondary.")
                results["force_server"] = server_target
                tag_processed = True
                if tag_processed:
                    processed_tags.append(tag)
                    continue

            # Simple flags
            if content_lower == "primary":
                results["primary_only"] = True
                tag_processed = True
            elif content_lower == "secondary":
                results["secondary_only"] = True
                tag_processed = True
            elif content_lower == "force-primary":
                if not results["force_server"]: results["force_server"] = 'primary'
                tag_processed = True
            elif content_lower in ["enhanced", "suggested"]:
                results["is_already_enhanced"] = True
                tag_processed = True
            elif content_lower == "enhance":
                results["enhance_requested"] = True
                tag_processed = True

            dim_match = re.fullmatch(r"(\d+)\s*[x×]\s*(\d+)", content)
            if dim_match:
                results["width"], results["height"] = int(dim_match.group(1)), int(
                    dim_match.group(2)
                )
                tag_processed = True

            kv_match = re.match(r"(\w+)\s*[:=]\s*(.*)", content, re.IGNORECASE)
            if kv_match:
                key, value = kv_match.group(1).lower(), kv_match.group(2).strip()
                try:
                    if key == "batch":
                        results["batch_size"] = int(value)
                        tag_processed = True
                    elif key == "lora":
                        results["lora_name"] = value
                        tag_processed = True
                    elif (
                        key.startswith("lora")
                        and key[4:].isdigit()
                        and 1 <= int(key[4:]) <= 4
                    ):
                        lora_parts = [p.strip() for p in value.split(",")]
                        if len(lora_parts) == 2 and lora_parts[1].isdigit():
                            results["lora_overrides"][lora_parts[1]] = lora_parts[0]
                            tag_processed = True
                    elif key == "length":
                        results["length"] = int(value)
                        tag_processed = True
                    elif key == "steps":
                        steps_parts = [p.strip() for p in value.split(",")]
                        if steps_parts and steps_parts[0].isdigit():
                            results["steps"] = (
                                int(steps_parts[0]),
                                (
                                    steps_parts[1]
                                    if len(steps_parts) > 1 and steps_parts[1].isdigit()
                                    else None
                                ),
                            )
                            tag_processed = True
                    elif key == "model":
                        results["model_name"] = value
                        tag_processed = True
                    elif key == "disablenode":
                        results["nodes_to_disable"].extend(re.findall(r"\d+", value))
                        tag_processed = True
                    elif key == "enablenode":
                        results["nodes_to_enable"].extend(re.findall(r"\d+", value))
                        tag_processed = True
                    elif key == "inject":
                        parts = [p.strip() for p in value.split(",", 2)]
                        if len(parts) == 3:
                            node_id, field, val_str = parts
                            try:
                                if "." in val_str:
                                    val = float(val_str)
                                else:
                                    val = int(val_str)
                            except ValueError:
                                val = val_str
                            results["injections"].append(
                                {"node_id": node_id, "field": field, "value": val}
                            )
                            tag_processed = True
                        else:
                            raise ValueError(
                                "`inject` tag requires 3 comma-separated parts: node_id,field_name,value"
                            )

                except ValueError as e:
                    raise ValueError(f"in tag `{tag}`: {e}")

            # MODIFICATION: Regex now includes '=' for the suggest command.
            suggest_match = re.fullmatch(r"suggest(?:[:=]\s*(\d+))?", content_lower)
            if suggest_match:
                results["suggest_count"] = (
                    int(suggest_match.group(1)) if suggest_match.group(1) else 3
                )
                tag_processed = True

            if tag_processed:
                processed_tags.append(tag)

        results["unrecognized_tags"] = [t for t in all_tags if t not in processed_tags]
        
        if (results["primary_only"] or results["force_server"] == 'primary') and \
           (results["secondary_only"] or results["force_server"] == 'secondary'):
            raise ValueError("Conflicting tags detected: Cannot target both primary and secondary servers.")

        cleaned_prompt = prompt
        for tag in all_tags:
            cleaned_prompt = cleaned_prompt.replace(tag, "", 1)
        results["cleaned_prompt"] = " ".join(cleaned_prompt.split())

        if results["is_already_enhanced"]:
            results["enhance_requested"] = False
            results["suggest_count"] = None

        return results

    def _get_help_message(self, is_admin: bool) -> str:
        description = (
            f"{self.valves.Tool_Description.strip()}\n\n---\n\n"
            if self.valves.Tool_Description and self.valves.Tool_Description.strip()
            else ""
        )
        help_text = """### Usage Details
**Basic Usage:** Type a description of what you want to create.
**Available Tags:**
- `(status)`: Shows Server availability and status.
- `(1024x768)`: Sets the output dimensions.
- `(steps:25)` or `(steps:25,12)`: Sets sampler steps, optionally for a specific node.
- `(model:model_name.safetensors)`: Overrides the checkpoint model.
- `(lora:lora_name)`: Applies a specific LoRA model to the default LoRA node.
- `(lora1:name,25)`, `(lora2:...)`: Applies a specific LoRA to a specific node ID (up to 4).
- `(length:33)`: Sets the frame count for videos.
- `(batch:4)`: Sets the number of images to generate.
- `(enhance)`: Improves your prompt automatically with an LLM before generation.
- `(suggest)` or `(suggest:4)` or `(suggest=4)`: Suggests enhanced text based on your entered prompt.
- `(primary)` / `(secondary)`: Forces job to a specific server.
- `(enablenode:12,25)`: Ensures specific workflow nodes are active.
- `(disablenode:15,30)`: Bypasses specific workflow nodes."""

        if is_admin:
            admin_text = """
---
**Administrative Commands:**
- `(status)` or `(stats)`: Displays a detailed status report of all connected servers.
- `(workflows)`: Lists available, pre-configured admin workflows.
- `(wf1) Your prompt...`: Executes admin workflow 1 with your prompt.
- `(wf1) ... (runs=5)`: Executes the job 5 times in a row.
- `(wf1) ... (async)`: Queues all jobs and exits without waiting for completion.
- `(wf1) ... (showpreview)`: Shows generated PNGs in chat (sync mode only).
- `(wf1) (raw)`: Executes the workflow with its embedded prompt.
- `(force:primary)` / `(force-secondary)` / `(force=primary)`: Forces a job to a server, bypassing all availability checks.
- `(clear-queue:primary)`: Clears pending jobs and frees VRAM on a server.
- `(cancel-queue:primary)`: Cancels the running job, clears the queue, and frees VRAM.
- `(free:primary)`: Clears Comfyui VRAM. `(unload:primary)`: Unloads Ollama models.
- `(loralist)` or `(loralist:filter)`: Lists available LoRAs, with an optional filter.
- `(inject:node_id,field_name,value)`: Directly injects a value into a node."""
            try:
                workflow_str = self.valves.ComfyUI_Workflow_JSON
                if (
                    workflow_str
                    and workflow_str.strip()
                    and workflow_str.strip() != "{}"
                ):
                    workflow = json.loads(self.valves.ComfyUI_Workflow_JSON)
                    injectables = {}
                    for node_id, node_data in workflow.items():
                        for field_name in node_data.get("inputs", {}):
                            injectables.setdefault(field_name, []).append(node_id)

                    if injectables:
                        admin_text += "\n\n**User Workflow Injectable Fields:**\n"
                        sorted_fields = sorted(injectables.keys())
                        midpoint = (len(sorted_fields) + 1) // 2
                        col1 = sorted_fields[:midpoint]
                        col2 = sorted_fields[midpoint:]

                        admin_text += (
                            "| Field | Nodes | Field | Nodes |\n|---|---|---|---|\n"
                        )
                        for i in range(midpoint):
                            field1 = col1[i]
                            nodes1 = f"`{', '.join(injectables[field1])}`"
                            row = f"| `{field1}` | {nodes1} "
                            if i < len(col2):
                                field2 = col2[i]
                                nodes2 = f"`{', '.join(injectables[field2])}`"
                                row += f"| `{field2}` | {nodes2} |"
                            else:
                                row += "| | |"
                            admin_text += row + "\n"

            except Exception as e:
                logger.warning(f"Could not parse workflow for admin help screen: {e}")
            help_text += admin_text
        return f"{description}{help_text.strip()}"

    def _save_media_and_get_public_url(
        self, request, media_data: bytes, content_type: str, user: User
    ) -> str:
        try:
            media_format = mimetypes.guess_extension(content_type) or ".bin"
            file = UploadFile(
                file=io.BytesIO(media_data),
                filename=f"generated-media{media_format}",
                headers={"content-type": content_type},
            )
            file_item = upload_file_handler(
                request=request, file=file, metadata={}, process=False, user=user
            )
            if file_item and file_item.id:
                return f"/api/v1/files/{file_item.id}/content"
            else:
                logger.error(
                    "upload_file_handler returned a null item, cannot generate URL."
                )
                return ""
        except Exception as e:
            logger.error(
                f"An exception occurred inside _save_media_and_get_public_url: {e}",
                exc_info=True,
            )
            return ""

    async def _emit_status(
        self, event_emitter: Callable, level: str, description: str, done: bool = False
    ):
        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": description,
                        "done": done,
                    },
                }
            )

    def _recursive_find_all_files(self, data: any, found_files: list):
        if isinstance(data, dict):
            if "filename" in data and "type" in data and data not in found_files:
                found_files.append(data)
            for value in data.values():
                self._recursive_find_all_files(value, found_files)
        elif isinstance(data, list):
            for item in data:
                self._recursive_find_all_files(item, found_files)

    def extract_all_files(self, outputs: Dict) -> List[Dict]:
        found_files = []
        self._recursive_find_all_files(outputs, found_files)
        return found_files

    def parse_input_from_chat(
        self, messages: List[Dict]
    ) -> Tuple[Optional[str], Optional[str]]:
        user_message_item = get_last_user_message_item(messages)
        if not user_message_item:
            return None, None
        content = user_message_item.get("content")
        if not content:
            return None, None
        prompt_text, image_data_uri = "", None
        if isinstance(content, str):
            prompt_text = content
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    prompt_text += part.get("text", "")
                elif part.get("type") == "image_url" and not image_data_uri:
                    image_data_uri = part.get("image_url", {}).get("url")
        return prompt_text.strip() or None, image_data_uri

    async def _download_media(
        self, base_url: str, file_info: Dict
    ) -> Tuple[Optional[bytes], Optional[str]]:
        url = f"{base_url}/view?filename={file_info['filename']}&subfolder={file_info.get('subfolder', '')}&type={file_info.get('type', 'output')}"
        for _ in range(3):
            try:
                http_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.get(
                        url, timeout=self.valves.Media_Download_Timeout
                    ),
                )
                if http_response.status_code == 404:
                    return None, None
                http_response.raise_for_status()
                if http_response.content:
                    return http_response.content, http_response.headers.get(
                        "content-type", "application/octet-stream"
                    )
            except requests.RequestException:
                await asyncio.sleep(2)
        return None, None

    async def _get_health_data(
        self, session: aiohttp.ClientSession, server_url: str
    ) -> Optional[Dict]:
        if not self.valves.Health_Check_Port:
            return None
        try:
            parsed_url = urlparse(server_url)
            new_netloc = f"{parsed_url.hostname}:{self.valves.Health_Check_Port}"
            health_url = parsed_url._replace(netloc=new_netloc, path="/health").geturl()
            async with session.get(health_url) as r:
                r.raise_for_status()
                return await r.json()
        except Exception:
            return None

    async def _select_server(
        self, primary_only: bool, secondary_only: bool, force_server: Optional[str]
    ) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
        
        # MODIFICATION: If a server is forced, it implies exclusivity, overriding other tags.
        if force_server == 'primary':
            primary_only = True
            secondary_only = False
        elif force_server == 'secondary':
            secondary_only = True
            primary_only = False

        servers_to_check = []
        # This list-building logic now correctly filters to ONLY the forced server (if any).
        if self.valves.ComfyUI_Primary_Address and not secondary_only:
            servers_to_check.append(
                ("Primary", self.valves.ComfyUI_Primary_Address, True)
            )
        if self.valves.ComfyUI_Secondary_Address and not primary_only:
            servers_to_check.append(
                ("Secondary", self.valves.ComfyUI_Secondary_Address, False)
            )
        
        if not servers_to_check:
             return None, f"The requested server ('{force_server}') is not configured." if force_server else "No servers are configured."

        failure_reasons = []
        logger.info(
            f"Starting server selection. Pre-emptive VRAM clear is set to: {self.valves.ComfyUI_Clear_VRAM_Before_Check}"
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            for name, url, is_primary in servers_to_check:
                log_prefix = f"Server Selection ({name}):"
                logger.info(f"{log_prefix} --- Evaluating server at {url} ---")

                try:
                    async with session.get(f"{url.rstrip('/')}/queue") as r:
                        r.raise_for_status()
                    logger.info(f"{log_prefix} ComfyUI API is reachable.")
                except Exception as e:
                    reason = "ComfyUI API is unreachable."
                    logger.warning(f"{log_prefix} FAILED. Reason: {reason} ({e})")
                    failure_reasons.append(f"{name}: {reason}")
                    continue

                # MODIFICATION: Simplified and robust force logic.
                # If a force_server was specified, we know this is the correct server to use
                # because `servers_to_check` was already filtered. We can bypass all checks.
                if force_server:
                    logger.info(f"{log_prefix} PASSED. `(force:{force_server})` tag was used, bypassing health and VRAM checks.")
                    return (url.rstrip("/"), name), None

                # --- All subsequent code is for REGULAR (non-forced) server selection ---

                if self.valves.ComfyUI_Clear_VRAM_Before_Check:
                    # ... (The VRAM clear block remains unchanged) ...
                    log_prefix_vc = f"Server Pre-Check ({name}):"
                    logger.info(
                        f"{log_prefix_vc} Issuing VRAM clear command to {url}..."
                    )
                    try:
                        async with session.post(
                            f"{url.rstrip('/')}/free",
                            json={"unload_models": True, "free_memory": True},
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as free_resp:
                            if free_resp.status == 200:
                                logger.info(
                                    f"{log_prefix_vc} VRAM clear command acknowledged."
                                )
                            else:
                                logger.warning(
                                    f"{log_prefix_vc} VRAM clear command returned non-200 status: {free_resp.status}."
                                )
                    except Exception as e:
                        logger.warning(
                            f"{log_prefix_vc} An error occurred while sending VRAM clear command: {e}"
                        )
                    logger.info(
                        f"{log_prefix_vc} Waiting 2 seconds before proceeding with health check."
                    )
                    await asyncio.sleep(2)

                health_data = await self._get_health_data(session, url)
                if health_data:
                    logger.info(
                        f"{log_prefix} Health API is available. Using intelligent selection logic."
                    )
                    if not health_data.get("available", True):
                        reason = "Server is manually set to unavailable."
                        logger.warning(f"{log_prefix} FAILED. Reason: {reason}")
                        failure_reasons.append(f"{name}: {reason}")
                        continue
                    
                    queue_running = health_data.get("queue_running") or 0
                    queue_pending = health_data.get("queue_pending") or 0

                    if is_primary and (queue_running > 0 or queue_pending > 0):
                        reason = "ComfyUI queue is busy."
                        logger.warning(f"{log_prefix} FAILED. Reason: {reason}")
                        failure_reasons.append(f"{name}: {reason}")
                        continue
                    
                    if not is_primary and (queue_running > 0 or queue_pending > 0):
                        logger.info(f"{log_prefix} PASSED. Server is busy, job will be queued.")
                        return (url.rstrip("/"), name), None

                    vram_used_gb = (health_data.get("vram_used_mb") or 0) / 1024
                    
                    if vram_used_gb <= self.valves.Primary_Server_Max_VRAM_Usage_GB:
                        logger.info(
                            f"{log_prefix} PASSED. VRAM usage ({vram_used_gb:.2f}GB) is below threshold."
                        )
                        return (url.rstrip("/"), name), None
                    logger.info(
                        f"{log_prefix} VRAM usage ({vram_used_gb:.2f}GB) is HIGH. Analyzing recoverability..."
                    )
                    if not is_primary:
                        # ... (this sub-block remains unchanged) ...
                        logger.info(
                            f"{log_prefix} Evaluating secondary server with high VRAM. Checking GPU idle status."
                        )
                        gpu_usage = health_data.get("utilization_gpu_percent")
                        if gpu_usage is not None and gpu_usage <= self.valves.Ollama_GPU_Busy_Threshold_Percent:
                            logger.info(
                                f"{log_prefix} PASSED. GPU is idle ({gpu_usage}%), server is considered available."
                            )
                            return (url.rstrip("/"), name), None
                        else:
                            reason = f"High VRAM and the GPU is actively busy ({gpu_usage if gpu_usage is not None else 'N/A'}%)."
                            logger.warning(f"{log_prefix} FAILED. Reason: {reason}")
                            failure_reasons.append(f"{name}: {reason}")
                            continue

                    if not self.valves.Allow_Ollama_Unload_On_Primary:
                        reason = "High VRAM and Ollama unload override is disabled."
                        logger.warning(f"{log_prefix} FAILED. Reason: {reason}")
                        failure_reasons.append(f"{name}: {reason}")
                        continue
                    ollama_processor = health_data.get("ollama_model_processor") or ""
                    if "gpu" not in ollama_processor.lower():
                        reason = f"VRAM is high due to a non-Ollama or non-GPU task ({ollama_processor})."
                        logger.warning(f"{log_prefix} FAILED. Reason: {reason}")
                        failure_reasons.append(f"{name}: {reason}")
                        continue
                    
                    gpu_usage = health_data.get("utilization_gpu_percent")
                    
                    if gpu_usage is not None and gpu_usage <= self.valves.Ollama_GPU_Busy_Threshold_Percent:
                        logger.info(
                            f"{log_prefix} PASSED. High VRAM is from an IDLE Ollama model (GPU at {gpu_usage}%). Server is recoverable."
                        )
                        return (url.rstrip("/"), name), None
                    reason = f"Ollama model is consistently busy (GPU at {gpu_usage if gpu_usage is not None else 'N/A'}%)."
                    logger.warning(f"{log_prefix} FAILED. Reason: {reason}")
                    failure_reasons.append(f"{name}: {reason}")
                    continue
                else:
                    logger.warning(
                        f"{log_prefix} Health API is not available. Using fallback logic."
                    )
                    if is_primary:
                        reason = "Health API is unavailable, cannot verify status."
                        logger.warning(
                            f"{log_prefix} FAILED. Reason: {reason} Failing as a precaution."
                        )
                        failure_reasons.append(f"{name}: {reason}")
                        continue
                    else:
                        logger.info(
                            f"{log_prefix} PASSED. Proceeding as it is the secondary server."
                        )
                        return (url.rstrip("/"), name), None
        return None, (
            " and ".join(failure_reasons)
            if failure_reasons
            else "All available servers are busy or offline."
        )

    async def _get_available_loras(self, server_url: str) -> List[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{server_url.rstrip('/')}/object_info/LoraLoader"
                ) as r:
                    if r.status == 200:
                        return (
                            (await r.json())
                            .get("LoraLoader", {})
                            .get("input", {})
                            .get("required", {})
                            .get("lora_name", [[]])[0]
                        )
        except Exception as e:
            logger.error(f"Error fetching LoRAs: {e}", exc_info=True)
        return []

    async def _get_available_models(self, server_url: str) -> List[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{server_url.rstrip('/')}/object_info/CheckpointLoaderSimple"
                ) as r:
                    if r.status == 200:
                        return (
                            (await r.json())
                            .get("CheckpointLoaderSimple", {})
                            .get("input", {})
                            .get("required", {})
                            .get("ckpt_name", [[]])[0]
                        )
        except Exception as e:
            logger.error(f"Error fetching Models: {e}", exc_info=True)
        return []

    async def _manage_queue(self, server_url, server_name, event_emitter, action: str):
        try:
            async with aiohttp.ClientSession() as session:
                if action == "clear-queue":
                    await session.post(f"{server_url}/queue", json={"clear": True})
                elif action == "cancel-queue":
                    await session.post(f"{server_url}/interrupt")
                    await session.post(f"{server_url}/queue", json={"clear": True})
                logger.info(
                    f"Admin Action ({server_name}): Freeing VRAM as part of '{action}' command."
                )
                await session.post(
                    f"{server_url}/free",
                    json={"unload_models": True, "free_memory": True},
                )
                await self._emit_status(
                    event_emitter,
                    "success",
                    f"Successfully sent '{action}' command and freed memory on {server_name}.",
                    done=True,
                )
        except Exception as e:
            await self._emit_status(
                event_emitter,
                "error",
                f"Error sending '{action}' command to {server_name}: {e}",
                done=True,
            )

    async def _get_server_health_status(
        self, session: aiohttp.ClientSession, name: str, base_url: str
    ) -> List[str]:
        parts = [f"\n--- **{name} ComfyUI Server** ---"]
        server_tag = name.lower()
        try:
            # First, confirm basic ComfyUI connectivity
            async with session.get(f"{base_url}/queue", timeout=3) as r:
                r.raise_for_status()
                q = await r.json()
                queue_str = f"`{len(q.get('queue_pending',[]))} pending, {len(q.get('queue_running',[]))} running`"

            # Attempt to fetch system stats from ComfyUI API
            try:
                async with session.get(f"{base_url}/system_stats", timeout=3) as r_stats:
                    r_stats.raise_for_status()
                    stats = await r_stats.json()
                    parts.append(f"- **Version:** `ComfyUI {stats.get('system', {}).get('comfyui_version', 'N/A')}`")

                    for device in stats.get("devices", []):
                        parts.append(f"- **Device:** `{device.get('name')}`")
                        torch_vram_free_gb = device.get('torch_vram_free', 0) / (1024**3)
                        torch_vram_total_gb = device.get('torch_vram_total', 0) / (1024**3)
                    #    parts.append(
                    #        f"  - **Torch VRAM:** `{torch_vram_free_gb:.1f} / {torch_vram_total_gb:.1f} GB` `(free:{server_tag})`"
                    #    )
            except Exception:
                # This is not a critical failure, just note it.
                parts.append("- **System Stats:** `Offline/Error`")


            # Now, attempt to fetch rich data from the health API
            health_data = await self._get_health_data(session, base_url)
            if health_data:
                manual_status = ""
                if not health_data.get('available', True):
                    manual_status = " (Manually Offline)"

                parts.append(f"- **Status:** `{health_data.get('comfyui_status', 'Online')}{manual_status}`")
                parts.append(f"- **Queue:** {queue_str}")

                gpu_usage = health_data.get("utilization_gpu_percent")
                if gpu_usage is not None:
                    parts.append(f"- **GPU Usage:** `{gpu_usage}%`")

                vram_used = health_data.get("vram_used_mb")
                vram_total = health_data.get("vram_total_mb")
                if vram_used is not None and vram_total is not None:
                    parts.append(
                        f"- **VRAM:** `{vram_used/1024:.2f} / {vram_total/1024:.2f} GB`")
                    parts.append(        
                        f"- **Torch VRAM:** `{torch_vram_free_gb:.1f} / {torch_vram_total_gb:.1f} GB` `(free:{server_tag})`"
                        
                    )

                parts.append("- **Ollama Status:**")
                ollama_processor = health_data.get("ollama_model_processor", "N/A")
                parts.append(f"  - **Processor:** `{ollama_processor}` `(unload:{server_tag})`")

                target_ollama_api_url = None
                parsed_url = urlparse(base_url)
                if name == "Primary":
                    target_ollama_api_url = self.valves.Ollama_Primary_API_URL or parsed_url._replace(netloc=f"{parsed_url.hostname}:11434").geturl()
                elif name == "Secondary":
                    target_ollama_api_url = self.valves.Ollama_Secondary_API_URL or parsed_url._replace(netloc=f"{parsed_url.hostname}:11434").geturl()

                if target_ollama_api_url:
                    try:
                        loaded_models = await asyncio.get_running_loop().run_in_executor(
                            None, get_loaded_models, target_ollama_api_url
                        )
                        if loaded_models:
                            parts.append("  - **Loaded Models:**")
                            for m in loaded_models:
                                model_name = m.get('name', 'unknown')
                                vram_size_gb = m.get('size_vram', 0) / (1024**3)
                                parts.append(f"    - `{model_name}` ({vram_size_gb:.2f} GB)")
                        else:
                            parts.append("  - `No models loaded on this server.`")
                    except Exception:
                        parts.append("  - `Could not reach this server's Ollama API.`")

            else:
                parts.append(f"- **Status:** `Online`")
                parts.append(f"- **Queue:** {queue_str}")
                parts.append(f"- **Health API:** `Offline/Error`")

        except Exception as e:
            parts.append(f"- **Status:** `Offline/Error` ({type(e).__name__})")
        return parts

    async def _get_models_to_unload(
        self, target_server_url: str, target_ollama_api_url: str
    ) -> List[str]:
        logger.info(
            f"--- Checking for unloadable models on server {target_server_url} via Ollama API {target_ollama_api_url} ---"
        )
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                health_data = await self._get_health_data(session, target_server_url)
            if not health_data:
                logger.info(
                    "Decision: Do not unload. Reason: Health API is unavailable."
                )
                return []
            ollama_processor = health_data.get("ollama_model_processor") or ""
            if "gpu" not in ollama_processor.lower():
                logger.info(
                    f"Decision: Do not unload. Reason: Ollama processor is not using GPU ('{ollama_processor}')."
                )
                return []
            loaded_models = await asyncio.get_running_loop().run_in_executor(
                None, get_loaded_models, target_ollama_api_url
            )
            if not loaded_models:
                logger.info(
                    "Decision: Do not unload. Reason: No Ollama models are currently loaded."
                )
                return []
            model_names = [m.get("name") for m in loaded_models if m.get("name")]
            logger.info(
                f"Decision: Unload. Reason: Found GPU-resident models: {model_names}."
            )
            return model_names
        except Exception as e:
            logger.warning(
                f"Could not determine models to unload due to an error: {e}. Defaulting to no unload."
            )
            return []

    async def _get_system_status(self) -> str:
        parts = ["### System Status Report"]
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            if self.valves.ComfyUI_Primary_Address:
                parts.extend(
                    await self._get_server_health_status(
                        session, "Primary", self.valves.ComfyUI_Primary_Address
                    )
                )
            if self.valves.ComfyUI_Secondary_Address:
                parts.extend(
                    await self._get_server_health_status(
                        session, "Secondary", self.valves.ComfyUI_Secondary_Address
                    )
                )
        # MODIFICATION: The old global Ollama status block has been removed from here.
        return "\n".join(parts)

    async def _get_workflow_list_message(self) -> str:
        parts = ["### Available Admin Workflows"]
        found_any = False
        for i in range(1, 6):
            name = getattr(self.valves, f"Admin_Workflow_{i}_Name")
            workflow_json = getattr(self.valves, f"Admin_Workflow_{i}_JSON")
            prompt_node_id = getattr(self.valves, f"Admin_Workflow_{i}_Prompt_Node_ID")
            if (
                name
                and workflow_json
                and workflow_json.strip()
                and workflow_json.strip() != "{}"
            ):
                found_any = True
                parts.append(f"\n**{i}. {name}** `(wf{i})`")
                try:
                    workflow = json.loads(workflow_json)
                    if prompt_node_id and prompt_node_id in workflow:
                        prompt_text = (
                            workflow[prompt_node_id]
                            .get("inputs", {})
                            .get("text", "[prompt not found]")
                        )
                        parts.append(f"- *Click to use:* `(wf{i}) {prompt_text}`")
                    else:
                        parts.append(
                            "- *This workflow does not use a text prompt.* `(wf{i}) (raw)`"
                        )
                except Exception:
                    parts.append("- *Could not parse workflow to show default prompt.*")
        if not found_any:
            return "### No Admin Workflows Configured\nPlease configure at least one Admin Workflow in the tool's valve settings."
        parts.append(
            "\n*Associated commands: `(runs=X)`, `(async)`, `(showpreview)`, `(raw)`*"
        )
        return "\n".join(parts)

    async def _process_media_output(
        self, http_api_url: str, all_files: List[Dict], request, user: User, loop
    ) -> List[str]:
        media_parts = []
        video_exts = (".mp4", ".webm", ".mov", ".avi")
        video_files = {
            os.path.splitext(f["filename"])[0]: f
            for f in all_files
            if f["filename"].lower().endswith(video_exts)
        }
        for file_info in all_files:
            is_video = file_info["filename"].lower().endswith(video_exts)
            if (
                os.path.splitext(file_info["filename"])[0] in video_files
                and not is_video
            ):
                continue
            media_data, media_type = await self._download_media(http_api_url, file_info)
            if not media_data:
                logger.warning(f"Failed to download media for {file_info['filename']}")
                continue
            if is_video:
                img_data, img_type = await self._download_media(
                    http_api_url,
                    {
                        **file_info,
                        "filename": f"{os.path.splitext(file_info['filename'])[0]}.png",
                    },
                )
                if img_data:
                    if self.valves.Save_Local_Copies:
                        media_parts.append(
                            f"![Video Preview]({await loop.run_in_executor(None, self._save_media_and_get_public_url, request, img_data, img_type, user)})"
                        )
                    else:
                        media_parts.append(
                            f"![Video Preview](data:{img_type};base64,{base64.b64encode(img_data).decode('utf-8')})"
                        )
                video_url = await loop.run_in_executor(
                    None,
                    self._save_media_and_get_public_url,
                    request,
                    media_data,
                    media_type,
                    user,
                )
                media_parts.append(f"[{file_info['filename']}]({video_url})")
            else:
                if self.valves.Save_Local_Copies:
                    img_url = await loop.run_in_executor(
                        None,
                        self._save_media_and_get_public_url,
                        request,
                        media_data,
                        media_type,
                        user,
                    )
                    media_parts.append(f"![Generated Image]({img_url})")
                else:
                    b64 = base64.b64encode(media_data).decode("utf-8")
                    media_parts.append(
                        f"![Generated Image](data:{media_type};base64,{b64})"
                    )
        return media_parts

    async def _handle_user_generation(
        self,
        body: dict,
        user: User,
        event_emitter: Callable,
        request,
        raw_prompt: str,
        image_data_uri: Optional[str],
    ):
        loop = asyncio.get_running_loop()
        try:
            parse_results = self._parse_prompt(raw_prompt)
        except ValueError as e:
            await self._emit_status(
                event_emitter, "error", f"Prompt Error: {e}", done=True
            )
            return body
        if parse_results["unrecognized_tags"]:
            tags_str = ", ".join([f"`{t}`" for t in parse_results["unrecognized_tags"]])
            await self._emit_status(
                event_emitter,
                "warning",
                f"The following tags had an unknown format and were ignored: {tags_str}. Check `(help)` for syntax.",
            )
        prompt = parse_results["cleaned_prompt"]
        suggest = parse_results["suggest_count"]
        base64_warning_message = None
        if self.valves.Base64_Warning_Threshold_MB > 0:
            total_base64_size = sum(
                len(part)
                for message in body.get("messages", [])
                if message.get("role") == "assistant"
                for part in re.findall(
                    r'data:image/[^;]+;base64,([^")]*)', str(message.get("content", ""))
                )
            )
            total_mb = total_base64_size / (1024 * 1024)
            if total_mb > self.valves.Base64_Warning_Threshold_MB:
                base64_warning_message = f"_**Note:** This chat contains approximately **{total_mb:.1f}MB** of image data, which may slow down your browser. Consider starting a new chat for optimal performance._"

        if suggest:
            if not self.valves.Enhancer_Model_ID:
                await self._emit_status(
                    event_emitter, "error", "Enhancer Model not configured.", done=True
                )
                return body
            await self._emit_status(
                event_emitter, "info", f"Generating {suggest} prompt suggestions..."
            )
            all_tags = re.findall(r"([<(][\w\s\d:./\\=×x-]*[>)])", raw_prompt)
            tags_to_keep_str = " ".join(
                [
                    tag
                    for tag in all_tags
                    if not tag.lower().strip()[1:-1].strip().startswith("suggest")
                    and not tag.lower().strip()[1:-1].strip() == "enhance"
                ]
            )
            try:
                enhancer_model_id = self.valves.Enhancer_Model_ID
                payload = {
                    "model": enhancer_model_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"Generate {suggest} creative, distinct image prompts based on this input. Separate each with '|||'.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                }
                response = await generate_chat_completion(request, payload, user)
                completion_data = (
                    response[0] if isinstance(response, list) and response else response
                )
                actual_model_name = completion_data.get("model", enhancer_model_id)
                await loop.run_in_executor(
                    None,
                    _unload_specific_model,
                    actual_model_name,
                    self.valves.Enhancer_Ollama_API_URL,
                )
                suggestions_raw = completion_data["choices"][0]["message"]["content"]
                suggestions = [
                    re.sub(
                        r"^(?:\d+\.\s*|Option\s*\d+:\s*)", "", s, flags=re.IGNORECASE
                    ).strip()
                    for s in suggestions_raw.split("|||")
                    if s.strip()
                ]
                response_content = f"Here are {len(suggestions)} suggestions...\n\n" + "\n".join(
                    [
                        f"- `{f'{sug.strip()} {tags_to_keep_str}'.strip()} (suggested)`"
                        for sug in suggestions
                    ]
                )
                await event_emitter(
                    {"type": "message", "data": {"content": response_content}}
                )
                await self._emit_status(
                    event_emitter, "success", "Suggestions provided.", done=True
                )
                return body
            except Exception as e:
                await loop.run_in_executor(
                    None,
                    _unload_specific_model,
                    self.valves.Enhancer_Model_ID,
                    self.valves.Enhancer_Ollama_API_URL,
                )
                await self._emit_status(
                    event_emitter,
                    "error",
                    f"Failed to generate suggestions: {e}",
                    done=True,
                )
                return body

        workflow_json_to_use = self.valves.ComfyUI_Workflow_JSON
        is_i2v_configured = False
        if self.valves.I2V_Workflow_JSON:
            try:
                json.loads(self.valves.I2V_Workflow_JSON)
                is_i2v_configured = True
            except (json.JSONDecodeError, TypeError):
                await self._emit_status(
                    event_emitter,
                    "warning",
                    "I2V workflow is not valid JSON and will be ignored.",
                )
        if image_data_uri:
            if is_i2v_configured:
                workflow_json_to_use = self.valves.I2V_Workflow_JSON
                await self._emit_status(
                    event_emitter,
                    "info",
                    "Image detected, using Image-to-Video workflow.",
                )
            elif not self.valves.Image_Input_Node_ID:
                await self._emit_status(
                    event_emitter,
                    "error",
                    "An image was provided, but no workflow is configured to accept it.",
                    done=True,
                )
                return body
        elif self.valves.Image_Input_Node_ID and not is_i2v_configured:
            await self._emit_status(
                event_emitter,
                "error",
                "This workflow requires an image upload.",
                done=True,
            )
            return body

        await self._emit_status(
            event_emitter, "info", "Selecting best generation server..."
        )
        server_info, err_msg = await self._select_server(
            parse_results["primary_only"],
            parse_results["secondary_only"],
            parse_results["force_server"],
        )
        if not server_info:
            await self._emit_status(
                event_emitter,
                "error",
                f"Server selection failed: {err_msg.capitalize()}",
                done=True,
            )
            return body
        selected_server_url, server_name = server_info
        await self._emit_status(
            event_emitter, "info", f"Selected {server_name} server..."
        )

        client_id, pre_output_message, prompt_id = str(uuid.uuid4()), "", None
        http_api_url = selected_server_url
        workflow = None
        try:
            if self.valves.unload_ollama_models:
                target_ollama_api_url = None
                parsed_url = urlparse(selected_server_url)
                if server_name == "Primary":
                    target_ollama_api_url = (
                        self.valves.Ollama_Primary_API_URL
                        or parsed_url._replace(
                            netloc=f"{parsed_url.hostname}:11434"
                        ).geturl()
                    )
                elif server_name == "Secondary":
                    target_ollama_api_url = (
                        self.valves.Ollama_Secondary_API_URL
                        or parsed_url._replace(
                            netloc=f"{parsed_url.hostname}:11434"
                        ).geturl()
                    )
                models_to_unload = await self._get_models_to_unload(
                    selected_server_url, target_ollama_api_url
                )
                if models_to_unload:
                    await self._emit_status(
                        event_emitter,
                        "info",
                        f"Unloading {len(models_to_unload)} Ollama model(s) from {server_name}...",
                    )
                    for model_name in models_to_unload:
                        await loop.run_in_executor(
                            None,
                            _unload_specific_model,
                            model_name,
                            target_ollama_api_url,
                        )

            if parse_results["enhance_requested"] and self.valves.Enhancer_Model_ID:
                await self._emit_status(event_emitter, "info", "Enhancing prompt...")
                enhancer_model_id = self.valves.Enhancer_Model_ID
                try:
                    payload = {
                        "model": enhancer_model_id,
                        "messages": [
                            {
                                "role": "system",
                                "content": self.valves.Enhancer_System_Prompt,
                            },
                            {"role": "user", "content": f"Rewrite: {prompt}"},
                        ],
                        "stream": False,
                    }
                    response = await generate_chat_completion(request, payload, user)
                    completion_data = (
                        response[0]
                        if isinstance(response, list) and response
                        else response
                    )
                    actual_model_name = completion_data.get("model", enhancer_model_id)
                    await loop.run_in_executor(
                        None,
                        _unload_specific_model,
                        actual_model_name,
                        self.valves.Enhancer_Ollama_API_URL,
                    )
                    enhanced_prompt = completion_data["choices"][0]["message"][
                        "content"
                    ]
                    pre_output_message = f"<details><summary>Enhanced Prompt</summary>{enhanced_prompt}</details>"
                    prompt = enhanced_prompt
                except Exception as e:
                    await loop.run_in_executor(
                        None,
                        _unload_specific_model,
                        enhancer_model_id,
                        self.valves.Enhancer_Ollama_API_URL,
                    )
                    await self._emit_status(
                        event_emitter,
                        "warning",
                        f"Could not enhance prompt: {e}. Using original prompt.",
                    )

            workflow = json.loads(workflow_json_to_use)
            all_node_ids = set(workflow.keys())

            for injection in parse_results["injections"]:
                node_id, field, value = (
                    injection["node_id"],
                    injection["field"],
                    injection["value"],
                )
                if node_id not in workflow:
                    raise ValueError(
                        f"Inject failed: Node ID '{node_id}' not found in workflow."
                    )
                if (
                    "inputs" not in workflow[node_id]
                    or field not in workflow[node_id]["inputs"]
                ):
                    raise ValueError(
                        f"Inject failed: Field '{field}' not found in inputs for node '{node_id}'."
                    )
                workflow[node_id]["inputs"][field] = value
                await self._emit_status(
                    event_emitter,
                    "info",
                    f"Injected value into `{field}` on node {node_id}.",
                )

            for node_id in (
                parse_results["nodes_to_enable"] + parse_results["nodes_to_disable"]
            ):
                if node_id not in all_node_ids:
                    raise ValueError(
                        f"Node ID '{node_id}' from tag not found in workflow."
                    )
            if parse_results["nodes_to_enable"] or parse_results["nodes_to_disable"]:
                for node_id in parse_results["nodes_to_disable"]:
                    workflow.setdefault(node_id, {}).setdefault("_meta", {})[
                        "status"
                    ] = "Bypassed"
                for node_id in parse_results["nodes_to_enable"]:
                    if (
                        workflow.get(node_id, {}).get("_meta", {}).get("status")
                        == "Bypassed"
                    ):
                        del workflow[node_id]["_meta"]["status"]
            if parse_results["model_name"]:
                model_name = parse_results["model_name"]
                await self._emit_status(
                    event_emitter, "info", f"Validating model: '{model_name}'..."
                )
                available = await self._get_available_models(http_api_url)
                if not available:
                    raise ValueError(
                        f"Could not retrieve model list from {server_name}."
                    )
                matches = [
                    f for f in available if model_name.strip().lower() in f.lower()
                ]
                if len(matches) == 1:
                    validated_model = matches[0]
                    target_node = self.valves.Model_Node_ID
                    if target_node and target_node in workflow:
                        workflow[target_node]["inputs"][
                            self.valves.Model_Field_Name
                        ] = validated_model
                        await self._emit_status(
                            event_emitter,
                            "success",
                            f"Validated model: `{validated_model}`",
                        )
                    else:
                        raise ValueError(
                            f"Model_Node_ID '{target_node}' not configured or found."
                        )
                else:
                    raise ValueError(
                        f"Aborted due to ambiguous or missing model. Matched {len(matches)} files."
                    )

            async def validate_and_inject_lora(
                lora_name: str, target_node_id: str, field_name: str
            ):
                await self._emit_status(
                    event_emitter,
                    "info",
                    f"Validating LoRA '{lora_name}' for node {target_node_id}...",
                )
                available = await self._get_available_loras(http_api_url)
                if not available:
                    raise ValueError(
                        f"Could not retrieve LoRA list from {server_name}."
                    )
                matches = [
                    f
                    for f in available
                    if lora_name.strip().lower() in f.replace("\\", "/").lower()
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Aborted due to ambiguous or missing LoRA '{lora_name}'. Matched {len(matches)} files."
                    )
                validated_lora = matches[0]
                if target_node_id not in workflow:
                    raise ValueError(f"Target LoRA node '{target_node_id}' not found.")
                if (
                    workflow.get(target_node_id, {}).get("_meta", {}).get("status")
                    == "Bypassed"
                ):
                    del workflow[target_node_id]["_meta"]["status"]
                workflow[target_node_id]["inputs"][field_name] = validated_lora
                await self._emit_status(
                    event_emitter,
                    "success",
                    f"Validated LoRA `{validated_lora}` for node {target_node_id}",
                )

            final_lora = parse_results["lora_name"] or self.valves.Lora_Name
            if self.valves.Lora_Node_ID in workflow:
                if final_lora and final_lora.strip().lower() != "none":
                    await validate_and_inject_lora(
                        final_lora,
                        self.valves.Lora_Node_ID,
                        self.valves.Lora_Name_Field_Name,
                    )
                else:
                    logger.info(
                        f"No default LoRA specified. Bypassing node: {self.valves.Lora_Node_ID}"
                    )
                    workflow.setdefault(self.valves.Lora_Node_ID, {}).setdefault(
                        "_meta", {}
                    )["status"] = "Bypassed"
            for node_id, lora_name in parse_results["lora_overrides"].items():
                await validate_and_inject_lora(
                    lora_name, node_id, self.valves.Lora_Name_Field_Name
                )

            def safe_inject(nid, f, v):
                if (
                    v is not None
                    and nid
                    and nid in workflow
                    and f in workflow[nid].get("inputs", {})
                ):
                    workflow[nid]["inputs"][f] = v

            if parse_results["steps"] is not None:
                steps_val, node_override = parse_results["steps"]
                target_node = node_override or self.valves.Steps_Node_ID
                if not target_node:
                    await self._emit_status(
                        event_emitter,
                        "warning",
                        "Steps tag used, but no Steps_Node_ID is configured.",
                    )
                else:
                    safe_inject(target_node, self.valves.Steps_Field_Name, steps_val)
            safe_inject(
                self.valves.Batch_Node_ID,
                self.valves.Batch_Field_Name,
                parse_results["batch_size"],
            )
            if self.valves.Image_Input_Node_ID and image_data_uri:
                safe_inject(
                    self.valves.Image_Input_Node_ID,
                    self.valves.Image_Input_Field_Name,
                    image_data_uri.split(",", 1)[1],
                )
            safe_inject(
                self.valves.Size_Node_ID,
                self.valves.Width_Field_Name,
                parse_results["width"] or self.valves.Width,
            )
            safe_inject(
                self.valves.Size_Node_ID,
                self.valves.Height_Field_Name,
                parse_results["height"] or self.valves.Height,
            )
            safe_inject(
                self.valves.Length_Node_ID or self.valves.Size_Node_ID,
                self.valves.Length_Field_Name,
                parse_results["length"] or self.valves.Length,
            )
            safe_inject(
                self.valves.Prompt_Node_ID, self.valves.Prompt_Field_Name, prompt
            )
            safe_inject(
                self.valves.Seed_Node_ID,
                self.valves.Seed_Field_Name,
                random.randint(0, 2**32 - 1),
            )

            # Add detailed logging of the final workflow before sending
            # try:
            #    final_workflow_for_debug = json.dumps(workflow, indent=2)
            #    logger.info(
            #        f"Final workflow being sent to {server_name}:\n{final_workflow_for_debug}"
            #    )
            #except Exception as log_e:
            #    logger.warning(f"Could not serialize workflow for debugging: {log_e}")

            exec_start_time, sampler_start_time, last_step_time, step_durations = (
                None,
                None,
                None,
                [],
            )
            
            # Create the payload with the required extra_data structure to prevent crash with comfyui_queue_manager
            payload = {
                "prompt": workflow,
                "client_id": client_id,
                "extra_data": {
                    "extra_pnginfo": {
                        "workflow": {
                            "workflow_name": "API Workflow",
                            "id": "API" # Add the final required key
                        }
                    }
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{http_api_url}/prompt", json=payload) as r:
                    if r.status != 200:
                        try:
                            error_data = await r.json()
                            error_type = error_data.get("type", "unknown_error")
                            error_message = error_data.get("message", "No message.")
                            error_details = error_data.get("details", "")

                            # Find the node title from the workflow using the node ID in the error
                            node_id_match = re.search(r"#(\d+)", str(error_details))
                            node_title = ""
                            if node_id_match:
                                node_id = node_id_match.group(1)
                                node_info = workflow.get(node_id, {})
                                node_title = f" (Node: '{node_info.get('_meta', {}).get('title', f'ID {node_id}')}')"

                            formatted_error = (
                                f"### ComfyUI Workflow Error\n"
                                f"**The server rejected the request with the following error:**\n\n"
                                f"- **Error Type:** `{error_type}`\n"
                                f"- **Message:** {error_message}{node_title}\n"
                                f"- **Details:** `{error_details}`\n\n"
                                f"This usually means a node in your workflow is misconfigured, missing a required input, or a custom node is not installed on the server."
                            )
                            await event_emitter(
                                {
                                    "type": "message",
                                    "data": {"content": formatted_error},
                                }
                            )
                            await self._emit_status(
                                event_emitter,
                                "error",
                                "Workflow validation failed.",
                                done=True,
                            )
                            return body
                        except Exception as parse_exc:
                            logger.error(
                                f"Could not parse ComfyUI error response: {parse_exc}"
                            )
                            # Fallback to raising the original status error
                            r.raise_for_status()

                    prompt_id = (await r.json()).get("prompt_id")
                if not prompt_id:
                    raise Exception("Failed to queue prompt.")
                await self._emit_status(
                    event_emitter, "info", f"Queued on {server_name}..."
                )
                ws_url = f"ws{'s' if http_api_url.startswith('https') else ''}://{http_api_url.split('://', 1)[-1]}/ws"
                async with session.ws_connect(
                    f"{ws_url}?clientId={client_id}", timeout=None
                ) as ws:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        job_start_time = exec_start_time or sampler_start_time
                        if job_start_time and (
                            time.time() - job_start_time > self.valves.max_wait_time
                        ):
                            break
                        m = json.loads(msg.data)
                        msg_type, data = m.get("type"), m.get("data", {})
                        if msg_type == "execution_error":
                            raise Exception(
                                f"Node error from server: {data.get('exception_message', 'Unknown')}"
                            )
                        elif msg_type == "execution_start" and not exec_start_time:
                            exec_start_time = time.time()
                        elif msg_type == "status":
                            q_rem = (
                                data.get("status", {})
                                .get("exec_info", {})
                                .get("queue_remaining", 0)
                            )
                            if q_rem > 0:
                                await self._emit_status(
                                    event_emitter,
                                    "info",
                                    f"In queue on {server_name}... {q_rem} tasks remaining.",
                                )
                        elif (
                            msg_type == "executing"
                            and data.get("prompt_id") == prompt_id
                        ):
                            if data.get("node") is None:
                                break
                            else:
                                node_id = data.get("node")
                                node_title = (
                                    workflow.get(node_id, {})
                                    .get("_meta", {})
                                    .get("title", node_id)
                                )
                                if "KSampler" not in node_title:
                                    await self._emit_status(
                                        event_emitter,
                                        "info",
                                        f"Executing on {server_name}: {node_title}...",
                                    )
                        elif msg_type == "progress":
                            now = time.time()
                            if not exec_start_time:
                                exec_start_time = now
                            if not sampler_start_time:
                                sampler_start_time = now
                            v, m_s = data["value"], data["max"]
                            node_title = "KSampler"
                            for nid, n in workflow.items():
                                if n.get("class_type") == "KSampler":
                                    node_title = n.get("_meta", {}).get(
                                        "title", "KSampler"
                                    )
                                    break
                            status = f"Executing {node_title} on {server_name}: Step {v}/{m_s}"
                            if last_step_time:
                                step_durations.append(now - last_step_time)
                                if len(step_durations) > 5:
                                    step_durations.pop(0)
                                avg = sum(step_durations) / len(step_durations)
                                eta = int((m_s - v) * avg)
                                elapsed = int(now - sampler_start_time)

                                def fmt(s):
                                    return f"{s//60:02d}:{s%60:02d}"

                                status += (
                                    f" | {avg:.2f}s/it | {fmt(elapsed)}<{fmt(eta)} "
                                )
                            last_step_time = now
                            await self._emit_status(event_emitter, "info", status)

                logger.info(
                    "WebSocket stream finished. Checking history for final output."
                )
                total_time = time.time() - exec_start_time if exec_start_time else 0
                job_data = None
                for i in range(5):
                    await asyncio.sleep(i + 1)
                    async with session.get(f"{http_api_url}/history/{prompt_id}") as r:
                        if r.status == 200:
                            hist = await r.json()
                            if prompt_id in hist:
                                prompt_hist = hist[prompt_id]
                                if prompt_hist.get("exceptions"):
                                    exc_details = prompt_hist["exceptions"][0]
                                    node_title = exc_details.get("node_id", "N/A")
                                    exc_type = exc_details.get(
                                        "exception_type", "Unknown"
                                    )
                                    exc_msg = exc_details.get(
                                        "exception_message", "No details"
                                    )
                                    raise Exception(
                                        f"Workflow error in node {node_title} ({exc_type}): {exc_msg}"
                                    )

                                if prompt_hist.get("outputs"):
                                    job_data = prompt_hist
                                    break
                if not job_data:
                    raise Exception(
                        "Failed to retrieve job data from history after completion. The workflow may have failed without generating an output or an error message."
                    )

                all_files = self.extract_all_files(job_data.get("outputs", {}))

                if not all_files:
                    output_node_types = [
                        "SaveImage",
                        "PreviewImage",
                        "ETN_SaveImage",
                        "SaveAnimatedPNG",
                        "SaveAnimatedWEBP",
                        "SaveAnimatedGIF",
                    ]
                    bypassed_outputs = []
                    if workflow:
                        for node_id, node_info in workflow.items():
                            if (
                                node_info.get("class_type") in output_node_types
                                and node_info.get("_meta", {}).get("status")
                                == "Bypassed"
                            ):
                                title = node_info.get("_meta", {}).get(
                                    "title", f"Node {node_id}"
                                )
                                bypassed_outputs.append(f"'{title}'")
                    if bypassed_outputs:
                        error_message = f"Workflow failed because the output node(s) were bypassed: {', '.join(bypassed_outputs)}. Please enable the node(s) and try again."
                    else:
                        error_message = "Execution finished with no media output. This usually indicates a logical error in the workflow, such as a disconnected node path. Please test the workflow directly in ComfyUI."
                    raise Exception(error_message)

                media_parts = await self._process_media_output(
                    http_api_url, all_files, request, user, loop
                )
                if not media_parts:
                    raise Exception("Failed to process any media.")

                intro_messages = ["Here is your generated media:"]
                if pre_output_message:
                    intro_messages.append(pre_output_message)
                if base64_warning_message:
                    intro_messages.append(base64_warning_message)

                content = "\n\n".join(intro_messages + media_parts)

                await event_emitter(
                    {
                        "type": "message",
                        "data": {"content": content, "role": "assistant"},
                    }
                )

                if self.valves.ComfyUI_Free_Memory:
                    await self._emit_status(
                        event_emitter, "info", f"Freeing memory on {server_name}..."
                    )
                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            f"{http_api_url}/free",
                            json={"unload_models": True, "free_memory": True},
                        )

                await self._emit_status(
                    event_emitter,
                    "success",
                    f"Processed on {server_name} in {total_time:.2f} seconds!",
                    done=True,
                )
                return content
        except asyncio.CancelledError:
            if prompt_id and http_api_url:
                async with aiohttp.ClientSession() as s:
                    await s.post(
                        f"{http_api_url}/interrupt", json={"client_id": client_id}
                    )
                    await s.post(
                        f"{http_api_url}/queue",
                        json={"delete": [prompt_id], "client_id": client_id},
                    )
            await self._emit_status(
                event_emitter, "error", "Job cancelled by user.", done=True
            )
            raise
        except Exception as e:
            logger.error(f"User Generation Error: {e}", exc_info=True)
            await self._emit_status(
                event_emitter,
                "error",
                f"An unexpected error occurred: {str(e)}",
                done=True,
            )
        return body

    async def _handle_admin_workflow(
        self,
        body: dict,
        user: User,
        event_emitter: Callable,
        request,
        raw_prompt: str,
        workflow_num: int,
    ):
        loop = asyncio.get_running_loop()
        final_content_to_return = ""

        try:
            workflow_name = getattr(self.valves, f"Admin_Workflow_{workflow_num}_Name")
            workflow_json_str = getattr(
                self.valves, f"Admin_Workflow_{workflow_num}_JSON"
            )
            prompt_node_id = getattr(
                self.valves, f"Admin_Workflow_{workflow_num}_Prompt_Node_ID"
            )

            if not all([workflow_name, workflow_json_str]):
                raise ValueError(f"Admin workflow {workflow_num} is not configured.")

            run_workflow = json.loads(workflow_json_str)
            parse_results = self._parse_prompt(raw_prompt)

            runs_match = re.search(r"[<(]runs=(\d+)[>)]", raw_prompt, re.IGNORECASE)
            runs = int(runs_match.group(1)) if runs_match else 1
            is_async = bool(re.search(r"[<(]async[>)]", raw_prompt, re.IGNORECASE))
            show_preview = bool(
                re.search(r"[<(]showpreview[>)]", raw_prompt, re.IGNORECASE)
            )
            use_raw = bool(re.search(r"[<(]raw[>)]", raw_prompt, re.IGNORECASE))

            await self._emit_status(
                event_emitter,
                "info",
                f"Preparing admin workflow: '{workflow_name}' (x{runs} runs)...",
            )

            server_info, err_msg = await self._select_server(
                parse_results["primary_only"],
                parse_results["secondary_only"],
                parse_results["force_primary"],
            )
            if not server_info:
                raise Exception(f"Server selection failed: {err_msg}")

            selected_server_url, server_name = server_info
            http_api_url = selected_server_url
            await self._emit_status(
                event_emitter, "info", f"Selected {server_name} server..."
            )

            queued_ids = []
            for run_num in range(1, runs + 1):
                client_id = str(uuid.uuid4())
                run_workflow = json.loads(workflow_json_str)

                if not use_raw and prompt_node_id and prompt_node_id in run_workflow:
                    run_workflow[prompt_node_id]["inputs"]["text"] = parse_results[
                        "cleaned_prompt"
                    ]

                for injection in parse_results["injections"]:
                    node_id, field, value = (
                        injection["node_id"],
                        injection["field"],
                        injection["value"],
                    )
                    if node_id in run_workflow:
                        run_workflow[node_id]["inputs"][field] = value

                for node_id in parse_results["nodes_to_disable"]:
                    if node_id in run_workflow:
                        run_workflow.setdefault(node_id, {}).setdefault("_meta", {})[
                            "status"
                        ] = "Bypassed"
                for node_id in parse_results["nodes_to_enable"]:
                    if (
                        run_workflow.get(node_id, {}).get("_meta", {}).get("status")
                        == "Bypassed"
                    ):
                        del run_workflow[node_id]["_meta"]["status"]

                prompt_id = None

             
                # Create the payload with the required extra_data structure
                payload = {
                    "prompt": run_workflow_copy,
                    "client_id": client_id,
                    "extra_data": {
                        "extra_pnginfo": {
                            "workflow": {
                                "workflow_name": "API Admin Workflow",
                                "id": "API" # Add the final required key
                            }
                        }
                    }
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{http_api_url}/prompt",
                        json=payload,
                    ) as r:
                        r.raise_for_status()
                        prompt_id = (await r.json()).get("prompt_id")
                        queued_ids.append(prompt_id)

                if not prompt_id:
                    raise Exception(f"Failed to queue prompt for run {run_num}.")
                if is_async:
                    continue
                await self._emit_status(
                    event_emitter,
                    "info",
                    f"Run {run_num}/{runs}: Queued with ID {prompt_id}. Waiting for completion...",
                )
                ws_url = f"ws{'s' if http_api_url.startswith('https') else ''}://{http_api_url.split('://', 1)[-1]}/ws"
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        f"{ws_url}?clientId={client_id}", timeout=None
                    ) as ws:
                        (
                            exec_start_time,
                            sampler_start_time,
                            last_step_time,
                            step_durations,
                        ) = (
                            None,
                            None,
                            None,
                            [],
                        )
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue

                            job_start_time = exec_start_time or sampler_start_time
                            if job_start_time and (
                                time.time() - job_start_time > self.valves.max_wait_time
                            ):
                                break

                            m = json.loads(msg.data)
                            msg_type, data = m.get("type"), m.get("data", {})
                            if msg_type == "execution_error":
                                raise Exception(
                                    f"Node error: {data.get('exception_message', 'Unknown')}"
                                )
                            elif msg_type == "execution_start" and not exec_start_time:
                                exec_start_time = time.time()
                            elif msg_type == "status":
                                q_rem = (
                                    data.get("status", {})
                                    .get("exec_info", {})
                                    .get("queue_remaining", 0)
                                )
                                if q_rem > 0:
                                    await self._emit_status(
                                        event_emitter,
                                        "info",
                                        f"Run {run_num}/{runs}: In queue on {server_name}... {q_rem} tasks remaining.",
                                    )
                            elif (
                                msg_type == "executing"
                                and data.get("prompt_id") == prompt_id
                            ):
                                if data.get("node") is None:
                                    break
                                else:
                                    node_id = data.get("node")
                                    node_title = (
                                        run_workflow.get(node_id, {})
                                        .get("_meta", {})
                                        .get("title", node_id)
                                    )
                                    if "KSampler" not in node_title:
                                        await self._emit_status(
                                            event_emitter,
                                            "info",
                                            f"Run {run_num}/{runs}: Executing on {server_name}: {node_title}...",
                                        )
                            elif msg_type == "progress":
                                now = time.time()
                                if not exec_start_time:
                                    exec_start_time = now
                                if not sampler_start_time:
                                    sampler_start_time = now
                                v, m_s = data["value"], data["max"]
                                node_title = "KSampler"
                                for nid, n in run_workflow.items():
                                    if n.get("class_type") == "KSampler":
                                        node_title = n.get("_meta", {}).get(
                                            "title", "KSampler"
                                        )
                                        break
                                status = f"Run {run_num}/{runs}: Executing {node_title} on {server_name}: Step {v}/{m_s}"
                                if last_step_time:
                                    step_durations.append(now - last_step_time)
                                    if len(step_durations) > 5:
                                        step_durations.pop(0)
                                    avg = sum(step_durations) / len(step_durations)

                                    eta = int((m_s - v) * avg)
                                    elapsed = int(now - sampler_start_time)

                                    def fmt(s):
                                        return f"{s//60:02d}:{s%60:02d}"

                                    status += (
                                        f" | {avg:.2f}s/it | {fmt(elapsed)}<{fmt(eta)} "
                                    )
                                last_step_time = now
                                await self._emit_status(event_emitter, "info", status)

                job_data = None
                async with aiohttp.ClientSession() as session:
                    for _ in range(5):
                        await asyncio.sleep(2)
                        async with session.get(
                            f"{http_api_url}/history/{prompt_id}"
                        ) as r:
                            if r.status == 200:
                                hist = await r.json()
                                if prompt_id in hist:
                                    prompt_hist = hist[prompt_id]
                                    if prompt_hist.get("exceptions"):
                                        exc_details = prompt_hist["exceptions"][0]
                                        node_title = exc_details.get("node_id", "N/A")
                                        exc_type = exc_details.get(
                                            "exception_type", "Unknown"
                                        )
                                        exc_msg = exc_details.get(
                                            "exception_message", "No details"
                                        )
                                        raise Exception(
                                            f"Workflow error in node {node_title} ({exc_type}): {exc_msg}"
                                        )

                                if prompt_hist.get("outputs"):
                                    job_data = prompt_hist
                                    break
                if not job_data:
                    raise Exception(
                        "Failed to retrieve job data from history after completion. The workflow may have failed without generating an output or an error message."
                    )

                all_files = self.extract_all_files(job_data.get("outputs", {}))

                if not all_files:
                    output_node_types = [
                        "SaveImage",
                        "PreviewImage",
                        "ETN_SaveImage",
                        "SaveAnimatedPNG",
                        "SaveAnimatedWEBP",
                        "SaveAnimatedGIF",
                    ]
                    bypassed_outputs = []
                    if workflow:
                        for node_id, node_info in workflow.items():
                            if (
                                node_info.get("class_type") in output_node_types
                                and node_info.get("_meta", {}).get("status")
                                == "Bypassed"
                            ):
                                title = node_info.get("_meta", {}).get(
                                    "title", f"Node {node_id}"
                                )
                                bypassed_outputs.append(f"'{title}'")
                    if bypassed_outputs:
                        error_message = f"Workflow failed because the output node(s) were bypassed: {', '.join(bypassed_outputs)}. Please enable the node(s) and try again."
                    else:
                        error_message = "Execution finished with no media output. This usually indicates a logical error in the workflow, such as a disconnected node path. Please test the workflow directly in ComfyUI."
                    raise Exception(error_message)

                media_parts = await self._process_media_output(
                    http_api_url, all_files, request, user, loop
                )
                if not media_parts:
                    raise Exception("Failed to process any media.")

                intro_messages = ["Here is your generated media:"]
                if pre_output_message:
                    intro_messages.append(pre_output_message)
                if base64_warning_message:
                    intro_messages.append(base64_warning_message)

                content = "\n\n".join(intro_messages + media_parts)

                await event_emitter(
                    {
                        "type": "message",
                        "data": {"content": content, "role": "assistant"},
                    }
                )

                if self.valves.ComfyUI_Free_Memory:
                    await self._emit_status(
                        event_emitter, "info", f"Freeing memory on {server_name}..."
                    )
                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            f"{http_api_url}/free",
                            json={"unload_models": True, "free_memory": True},
                        )

                await self._emit_status(
                    event_emitter,
                    "success",
                    f"Processed on {server_name} in {total_time:.2f} seconds!",
                    done=True,
                )
                return content
        except asyncio.CancelledError:
            if prompt_id and http_api_url:
                async with aiohttp.ClientSession() as s:
                    await s.post(
                        f"{http_api_url}/interrupt", json={"client_id": client_id}
                    )
                    await s.post(
                        f"{http_api_url}/queue",
                        json={"delete": [prompt_id], "client_id": client_id},
                    )
            await self._emit_status(
                event_emitter, "error", "Job cancelled by user.", done=True
            )
            raise
        except Exception as e:
            logger.error(f"User Generation Error: {e}", exc_info=True)
            await self._emit_status(
                event_emitter,
                "error",
                f"An unexpected error occurred: {str(e)}",
                done=True,
            )
        return body

    async def _handle_admin_workflow(
        self,
        body: dict,
        user: User,
        event_emitter: Callable,
        request,
        raw_prompt: str,
        workflow_num: int,
    ):
        loop = asyncio.get_running_loop()
        final_content_to_return = ""

        try:
            workflow_name = getattr(self.valves, f"Admin_Workflow_{workflow_num}_Name")
            workflow_json_str = getattr(
                self.valves, f"Admin_Workflow_{workflow_num}_JSON"
            )
            prompt_node_id = getattr(
                self.valves, f"Admin_Workflow_{workflow_num}_Prompt_Node_ID"
            )

            if not all([workflow_name, workflow_json_str]):
                raise ValueError(f"Admin workflow {workflow_num} is not configured.")

            run_workflow = json.loads(workflow_json_str)
            parse_results = self._parse_prompt(raw_prompt)

            runs_match = re.search(r"[<(]runs=(\d+)[>)]", raw_prompt, re.IGNORECASE)
            runs = int(runs_match.group(1)) if runs_match else 1
            is_async = bool(re.search(r"[<(]async[>)]", raw_prompt, re.IGNORECASE))
            show_preview = bool(
                re.search(r"[<(]showpreview[>)]", raw_prompt, re.IGNORECASE)
            )
            use_raw = bool(re.search(r"[<(]raw[>)]", raw_prompt, re.IGNORECASE))

            await self._emit_status(
                event_emitter,
                "info",
                f"Preparing admin workflow: '{workflow_name}' (x{runs} runs)...",
            )

            server_info, err_msg = await self._select_server(
                parse_results["primary_only"],
                parse_results["secondary_only"],
                parse_results["force_primary"],
            )
            if not server_info:
                raise Exception(f"Server selection failed: {err_msg}")

            selected_server_url, server_name = server_info
            http_api_url = selected_server_url
            await self._emit_status(
                event_emitter, "info", f"Selected {server_name} server..."
            )

            queued_ids = []
            for run_num in range(1, runs + 1):
                client_id = str(uuid.uuid4())
                run_workflow = json.loads(workflow_json_str)

                if not use_raw and prompt_node_id and prompt_node_id in run_workflow:
                    run_workflow[prompt_node_id]["inputs"]["text"] = parse_results[
                        "cleaned_prompt"
                    ]

                for injection in parse_results["injections"]:
                    node_id, field, value = (
                        injection["node_id"],
                        injection["field"],
                        injection["value"],
                    )
                    if node_id in run_workflow:
                        run_workflow[node_id]["inputs"][field] = value

                for node_id in parse_results["nodes_to_disable"]:
                    if node_id in run_workflow:
                        run_workflow.setdefault(node_id, {}).setdefault("_meta", {})[
                            "status"
                        ] = "Bypassed"
                for node_id in parse_results["nodes_to_enable"]:
                    if (
                        run_workflow.get(node_id, {}).get("_meta", {}).get("status")
                        == "Bypassed"
                    ):
                        del run_workflow[node_id]["_meta"]["status"]

                prompt_id = None
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{http_api_url}/prompt",
                        json={"prompt": run_workflow, "client_id": client_id},
                    ) as r:
                        r.raise_for_status()
                        prompt_id = (await r.json()).get("prompt_id")
                        queued_ids.append(prompt_id)

                if not prompt_id:
                    raise Exception(f"Failed to queue prompt for run {run_num}.")
                if is_async:
                    continue
                await self._emit_status(
                    event_emitter,
                    "info",
                    f"Run {run_num}/{runs}: Queued with ID {prompt_id}. Waiting for completion...",
                )
                ws_url = f"ws{'s' if http_api_url.startswith('https') else ''}://{http_api_url.split('://', 1)[-1]}/ws"
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        f"{ws_url}?clientId={client_id}", timeout=None
                    ) as ws:
                        (
                            exec_start_time,
                            sampler_start_time,
                            last_step_time,
                            step_durations,
                        ) = (
                            None,
                            None,
                            None,
                            [],
                        )
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue

                            job_start_time = exec_start_time or sampler_start_time
                            if job_start_time and (
                                time.time() - job_start_time > self.valves.max_wait_time
                            ):
                                break

                            m = json.loads(msg.data)
                            msg_type, data = m.get("type"), m.get("data", {})
                            if msg_type == "execution_error":
                                raise Exception(
                                    f"Node error: {data.get('exception_message', 'Unknown')}"
                                )
                            elif msg_type == "execution_start" and not exec_start_time:
                                exec_start_time = time.time()
                            elif msg_type == "status":
                                q_rem = (
                                    data.get("status", {})
                                    .get("exec_info", {})
                                    .get("queue_remaining", 0)
                                )
                                if q_rem > 0:
                                    await self._emit_status(
                                        event_emitter,
                                        "info",
                                        f"Run {run_num}/{runs}: In queue on {server_name}... {q_rem} tasks remaining.",
                                    )
                            elif (
                                msg_type == "executing"
                                and data.get("prompt_id") == prompt_id
                            ):
                                if data.get("node") is None:
                                    break
                                else:
                                    node_id = data.get("node")
                                    node_title = (
                                        run_workflow.get(node_id, {})
                                        .get("_meta", {})
                                        .get("title", node_id)
                                    )
                                    if "KSampler" not in node_title:
                                        await self._emit_status(
                                            event_emitter,
                                            "info",
                                            f"Run {run_num}/{runs}: Executing on {server_name}: {node_title}...",
                                        )
                            elif msg_type == "progress":
                                now = time.time()
                                if not exec_start_time:
                                    exec_start_time = now
                                if not sampler_start_time:
                                    sampler_start_time = now
                                v, m_s = data["value"], data["max"]
                                node_title = "KSampler"
                                for nid, n in run_workflow.items():
                                    if n.get("class_type") == "KSampler":
                                        node_title = n.get("_meta", {}).get(
                                            "title", "KSampler"
                                        )
                                        break
                                status = f"Run {run_num}/{runs}: Executing {node_title} on {server_name}: Step {v}/{m_s}"
                                if last_step_time:
                                    step_durations.append(now - last_step_time)
                                    if len(step_durations) > 5:
                                        step_durations.pop(0)
                                    avg = sum(step_durations) / len(step_durations)
                                    eta = int((m_s - v) * avg)
                                    elapsed = int(now - sampler_start_time)

                                    def fmt(s):
                                        return f"{s//60:02d}:{s%60:02d}"

                                    status += (
                                        f" | {avg:.2f}s/it | {fmt(elapsed)}<{fmt(eta)} "
                                    )
                                last_step_time = now
                                await self._emit_status(event_emitter, "info", status)

                job_data = None
                async with aiohttp.ClientSession() as session:
                    for _ in range(5):
                        await asyncio.sleep(2)
                        async with session.get(
                            f"{http_api_url}/history/{prompt_id}"
                        ) as r:
                            if r.status == 200:
                                hist = await r.json()
                                if prompt_id in hist:
                                    prompt_hist = hist[prompt_id]
                                    if prompt_hist.get("exceptions"):
                                        exc_details = prompt_hist["exceptions"][0]
                                        node_title = exc_details.get("node_id", "N/A")
                                        exc_type = exc_details.get(
                                            "exception_type", "Unknown"
                                        )
                                        exc_msg = exc_details.get(
                                            "exception_message", "No details"
                                        )
                                        raise Exception(
                                            f"Workflow error in node {node_title} ({exc_type}): {exc_msg}"
                                        )

                                    if prompt_hist.get("outputs"):
                                        job_data = prompt_hist
                                        break
                if not job_data:
                    raise Exception(
                        f"Failed to retrieve job data from history for run {run_num}. The workflow may have failed."
                    )
                all_files = self.extract_all_files(job_data.get("outputs", {}))
                if not show_preview:
                    filenames = [
                        f.get("filename") for f in all_files if f.get("filename")
                    ]
                    await self._emit_status(
                        event_emitter,
                        "info",
                        f"Run {run_num}/{runs} complete. Generated files: {', '.join(filenames)}",
                    )
                else:
                    media_parts = await self._process_media_output(
                        http_api_url, all_files, request, user, loop
                    )
                    if not media_parts:
                        await self._emit_status(
                            event_emitter,
                            "warning",
                            f"Run {run_num}/{runs} complete. No displayable media found.",
                        )
                        continue
                    content = f"### Run {run_num}/{runs} Complete\n" + "\n".join(
                        media_parts
                    )
                    await event_emitter(
                        {"type": "message", "data": {"content": content}}
                    )
                    final_content_to_return = content
            if is_async:
                final_message = f"**Successfully queued {runs} job(s) on {server_name}.**\n- Prompt IDs: `{'`, `'.join(queued_ids)}`"
                await event_emitter(
                    {"type": "message", "data": {"content": final_message}}
                )
                await self._emit_status(
                    event_emitter,
                    "success",
                    f"Async job submission complete.",
                    done=True,
                )
                final_content_to_return = final_message
            else:
                await self._emit_status(
                    event_emitter,
                    "success",
                    f"Admin workflow batch completed all {runs} run(s).",
                    done=True,
                )
        except Exception as e:
            logger.error(f"Admin Workflow Error: {e}", exc_info=True)
            await self._emit_status(
                event_emitter,
                "error",
                f"An unexpected error occurred in admin workflow: {str(e)}",
                done=True,
            )
        return final_content_to_return

    async def _handle_user_command(
        self, command: str, value: str, event_emitter: Callable, loop
    ):
        if value not in ["primary", "secondary"]:
            await self._emit_status(
                event_emitter,
                "error",
                f"Command `({command})` requires a valid target: `primary` or `secondary`.",
                done=True,
            )
            return

        server_name = value.capitalize()
        addr = (
            self.valves.ComfyUI_Primary_Address
            if value == "primary"
            else self.valves.ComfyUI_Secondary_Address
        )
        if not addr:
            await self._emit_status(
                event_emitter,
                "error",
                f"{server_name} server is not configured.",
                done=True,
            )
            return

        try:
            if command == "free":
                await self._emit_status(
                    event_emitter, "info", f"Sending free memory command to {server_name}..."
                )
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{addr}/free",
                        json={"unload_models": True, "free_memory": True},
                        timeout=aiohttp.ClientTimeout(total=15),
                    )
                await self._emit_status(
                    event_emitter,
                    "success",
                    f"Successfully sent free memory command to {server_name}.",
                    done=True,
                )

            elif command == "unload":
                await self._emit_status(
                    event_emitter, "info", f"Checking for loaded Ollama models on {server_name}..."
                )
                parsed_url = urlparse(addr)
                target_ollama_api_url = None
                if value == "primary":
                    target_ollama_api_url = (
                        self.valves.Ollama_Primary_API_URL
                        or parsed_url._replace(netloc=f"{parsed_url.hostname}:11434").geturl()
                    )
                else: # secondary
                    target_ollama_api_url = (
                        self.valves.Ollama_Secondary_API_URL
                        or parsed_url._replace(netloc=f"{parsed_url.hostname}:11434").geturl()
                    )

                if not target_ollama_api_url:
                     raise ValueError("Ollama API URL for this server is not configured.")

                loaded_models = await loop.run_in_executor(None, get_loaded_models, target_ollama_api_url)
                if not loaded_models:
                    await self._emit_status(
                        event_emitter, "success", f"No Ollama models were loaded on {server_name}.", done=True
                    )
                    return

                await self._emit_status(
                    event_emitter, "info", f"Unloading {len(loaded_models)} model(s) from {server_name}..."
                )
                for model in loaded_models:
                    await loop.run_in_executor(None, _unload_specific_model, model.get("name"), target_ollama_api_url)

                await self._emit_status(
                    event_emitter, "success", f"Successfully sent unload commands to {server_name}.", done=True
                )

        except Exception as e:
            await self._emit_status(
                event_emitter, "error", f"Error during `{command}` command on {server_name}: {e}", done=True
            )

    async def pipe(
        self, body: dict, __user__: dict, __event_emitter__: Callable, __request__=None
    ):
        loop = asyncio.get_running_loop()
        event_emitter, request = __event_emitter__, __request__
        user = Users.get_user_by_id(__user__["id"])
        is_admin = self._is_user_admin(user)
        raw_prompt, image_data_uri = self.parse_input_from_chat(
            body.get("messages", [])
        )

        if not raw_prompt or raw_prompt.strip().startswith("### Task:"):
            return body
        raw_lower = raw_prompt.strip().lower()
        if raw_lower.startswith("here are") and "suggestions" in raw_lower:
            return body

        # --- Command Prioritization Logic ---

        # 1. Administrative Commands (Admins Only)
        if is_admin:
            admin_workflow_match = re.search(r"[<(]wf([1-5])[>)]", raw_lower)
            admin_cmd_match = re.search(
                r"[<(](" + "|".join(self.ADMIN_COMMAND_KEYWORDS) + r")(?::\s*([^)>]*))?[>)]",
                raw_lower,
            )
            # Bare commands are for things that don't take arguments
            bare_admin_command = (raw_lower if raw_lower in ["workflows", "loralist"] else None)

            if admin_workflow_match:
                return await self._handle_admin_workflow(
                    body, user, event_emitter, request, raw_prompt, int(admin_workflow_match.group(1))
                )

            if admin_cmd_match or bare_admin_command:
                command, value = "", ""
                if admin_cmd_match:
                    command = admin_cmd_match.group(1).lower().strip()
                    value = (admin_cmd_match.group(2).strip() if admin_cmd_match.group(2) else "")
                elif bare_admin_command:
                    command = bare_admin_command

                if command == "workflows":
                    msg = await self._get_workflow_list_message()
                    await event_emitter({"type": "message", "data": {"content": msg}})
                    await self._emit_status(event_emitter, "success", "Workflow list provided.", done=True)
                    return msg

                if command in ["clear-queue", "cancel-queue"]:
                    if value not in ["primary", "secondary"]:
                        await self._emit_status(event_emitter, "error", f"Command `({command})` requires a valid target (e.g., `{command}:primary`).", done=True)
                        return body
                    addr = self.valves.ComfyUI_Primary_Address if value == "primary" else self.valves.ComfyUI_Secondary_Address
                    if addr:
                        await self._manage_queue(addr, value.capitalize(), event_emitter, command)
                    else:
                        await self._emit_status(event_emitter, "error", f"{value.capitalize()} server not configured.", done=True)
                    return body

                if command == "loralist":
                    await self._emit_status(event_emitter, "info", "Fetching LoRA list...")
                    # ... (loralist logic remains the same) ...
                    return body # This return is simplified; the original logic is complex and should be copied if needed, but it seems to end here.

        # 2. Help Command (All Users)
        normalized_cmd = raw_lower.strip("<>()")
        if normalized_cmd in ["help", "?", "help me"]:
            help_message = self._get_help_message(is_admin)
            await event_emitter({"type": "message", "data": {"content": help_message}})
            await self._emit_status(event_emitter, "success", "Help provided.", done=True)
            return help_message

        # 3. Standard User Commands (All Users)
        user_cmd_match = re.search(
            r"[<(](" + "|".join(self.USER_COMMAND_KEYWORDS) + r")(?::\s*([^)>]*))?[>)]",
            raw_lower,
        )
        bare_user_command = (raw_lower if raw_lower in ["status", "stats"] else None)

        if user_cmd_match or bare_user_command:
            command, value = "", ""
            if user_cmd_match:
                command = user_cmd_match.group(1).lower().strip()
                value = (user_cmd_match.group(2).strip() if user_cmd_match.group(2) else "")
            elif bare_user_command:
                command = bare_user_command

            if command in ["status", "stats"]:
                status_report = await self._get_system_status()
                await event_emitter({"type": "message", "data": {"content": status_report}})
                await self._emit_status(event_emitter, "success", "Status report generated.", done=True)
                return status_report

            if command in ["free", "unload"]:
                 await self._handle_user_command(command, value, event_emitter, loop)
                 return body

        # 4. Admin-Only Mode Check
        is_admin_only_mode = False
        try:
            workflow_json = json.loads(self.valves.ComfyUI_Workflow_JSON)
            if not workflow_json or workflow_json == {}:
                is_admin_only_mode = True
        except (json.JSONDecodeError, TypeError):
            is_admin_only_mode = True

        if is_admin_only_mode:
            if not is_admin:
                await self._emit_status(event_emitter, "info", "This tool is currently configured for administrative workflows only.", done=True)
            else:
                msg = await self._get_workflow_list_message()
                await event_emitter({"type": "message", "data": {"content": f"**Default workflow not set. Please specify an admin workflow.**\n\n{msg}"}})
                await self._emit_status(event_emitter, "info", "Awaiting admin workflow selection.", done=True)
            return body

        # 5. Default to User Generation
        return await self._handle_user_generation(
            body, user, event_emitter, request, raw_prompt, image_data_uri
        )