from __future__ import annotations

import json
import os
import sys
import time
import logging
from typing import Optional, Any, Dict, Type

from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from icd_codes.llm.base import BaseLLM, LLMResponse
from icd_codes.exception import CustomException
from icd_codes.logger import get_logger

logger = get_logger(__name__)


class AsyncOpenAIClient:
    """
    Async OpenAI client that mirrors the sync OpenAIClient but uses AsyncOpenAI.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        # max_connections: int = 100,
        # max_keepalive_connections: int = 20,
        # timeout_seconds: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        logger.info(f"Initializing AsyncOpenAIClient for model: {model_name}")

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            error_msg = "OpenAI API Key not found in arguments or environment variables."
            logger.error(error_msg)
            raise CustomException(error_msg, sys)

        try:
            # http_client = httpx.AsyncClient(
            #     limits=httpx.Limits(
            #         max_connections=max_connections,
            #         max_keepalive_connections=max_keepalive_connections,
            #     ),
            #     timeout=httpx.Timeout(timeout_seconds),
            # )
            self.client = AsyncOpenAI(
                api_key=resolved_api_key, 
                # http_client=http_client
                )
            logger.debug("Async OpenAI client instance created successfully.")
        except Exception as exc:
            logger.exception("Failed to initialize Async OpenAI client.")
            raise CustomException(exc, sys)

    async def _raw_generate(
        self,
        prompt: str,
        response_model: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        logger.debug(f"Starting async generation request for model '{self.model_name}'")
        logger.debug(f"Prompt payload: {prompt[:100]}...")

        try:
            # This internal method is now less likely to be used with the new chat completion flow
            # but is kept for potential backward compatibility.
            # It currently calls a non-standard 'responses.parse' which may need updating
            # if this flow is to be used in the future.
            payload = {
                "model": self.model_name,
                "input": [{"role": "user", "content": prompt}],
                **kwargs,
            }

            if response_model is not None:
                logger.debug(f"Applying response model: {response_model.__class__.__name__}")
                payload["text_format"] = response_model

            logger.debug(f"Final payload sent to OpenAI: {payload}")

            response = await self.client.responses.parse(**payload)

            if response is None:
                logger.error("API returned a None response object.")
                raise CustomException("Received empty response from OpenAI API", sys)

            message_content = getattr(response, "output_text", "")
            if message_content is None:
                logger.error("Choice message content is None.")
                raise CustomException("Response from OpenAI API has no content", sys)

            if not str(message_content).strip():
                logger.warning("Response content is an empty string.")
                raise CustomException("Response from OpenAI API contains empty text", sys)

            usage = getattr(response, "usage", None)
            usage_data = {
                "prompt_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }

            logger.debug(f"Successfully received response. Tokens used: {usage_data['total_tokens']}")

            return {
                "text": message_content,
                "usage": usage_data,
                "metadata": {
                    "provider": "openai",
                    "model": self.model_name,
                    "request_id": getattr(response, "_request_id", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                },
            }

        except Exception as exc:
            logger.error(f"Error during OpenAI API execution: {str(exc)}")
            logger.exception("Full traceback for API failure:")
            raise CustomException(exc, sys)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        start_time = time.perf_counter()
        logger.debug(f"Async LLM Dispatch: Provider={self.__class__.__name__}, Model={self.model_name}")

        try:
            result = await self._raw_generate(prompt, response_model=response_model, **kwargs)

            latency = (time.perf_counter() - start_time) * 1000
            raw_text = result.get("text", "")
            usage = result.get("usage", {})
            metadata = result.get("metadata", {})

            processed_content: Any = raw_text
            if response_model:
                logger.debug(f"Attempting to parse output into schema: {response_model.__name__}")
                try:
                    processed_content = BaseLLM._parse_response(raw_text, response_model)
                    logger.debug("Structured output parsing successful.")
                except (ValidationError, ValueError, TypeError) as exc:
                    logger.error(f"Schema validation failed for model {self.model_name}")
                    logger.debug(f"Failed raw_text: {raw_text}")
                    raise CustomException(exc, sys)

            if isinstance(raw_text, str):
                raw_response_str = raw_text
            elif isinstance(raw_text, BaseModel):
                raw_response_str = raw_text.model_dump_json()
            else:
                try:
                    raw_response_str = json.dumps(raw_text)
                except (TypeError, ValueError):
                    raw_response_str = str(raw_text)

            logger.debug(f"Generation complete. Latency: {latency:.2f}ms. Total Tokens: {usage.get('total_tokens', 0)}")

            return LLMResponse(
                content=processed_content,
                raw_response=raw_response_str,
                model_name=self.model_name,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=round(latency, 2),
                provider=self.__class__.__name__,
                metadata=metadata if isinstance(metadata, dict) else None,
            )

        except Exception as exc:
            logger.error(f"Exhausted all retry attempts for {self.model_name}")
            raise CustomException(exc, sys)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    async def generate_from_prompt(
        self,
        prompt: Dict[str, Any],
        inputs: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        start_time = time.perf_counter()
        logger.debug(f"Async LLM Dispatch (prompt): Provider={self.__class__.__name__}, Model={self.model_name}")

        try:
            payload = {
                "prompt": prompt,
                **kwargs,
            }

            if inputs:
                payload["input"] = inputs

            response = await self.client.responses.create(**payload)

            if response is None:
                logger.error("API returned a None response object.")
                raise CustomException("Received empty response from OpenAI API", sys)

            message_content = getattr(response, "output_text", "")
            if message_content is None:
                logger.error("Choice message content is None.")
                raise CustomException("Response from OpenAI API has no content", sys)

            if not str(message_content).strip():
                logger.warning("Response content is an empty string.")
                raise CustomException("Response from OpenAI API contains empty text", sys)

            usage = getattr(response, "usage", None)
            usage_data = {
                "prompt_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }

            latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"Generation complete. Latency: {latency:.2f}ms. Total Tokens: {usage_data['total_tokens']}")

            return LLMResponse(
                content=message_content,
                raw_response=str(message_content),
                model_name=self.model_name,
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                latency_ms=round(latency, 2),
                provider=self.__class__.__name__,
                metadata={
                    "provider": "openai",
                    "model": self.model_name,
                    "request_id": getattr(response, "_request_id", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                },
            )

        except Exception as exc:
            logger.error(f"Exhausted all retry attempts for {self.model_name}")
            raise CustomException(exc, sys)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    async def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        start_time = time.perf_counter()
        logger.debug(f"Async LLM Dispatch (messages): Provider={self.__class__.__name__}, Model={self.model_name}")

        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                **kwargs,
            }

            response = await self.client.chat.completions.create(**payload)

            if response is None:
                logger.error("API returned a None response object.")
                raise CustomException("Received empty response from OpenAI API", sys)

            if not response.choices:
                logger.error("Response from OpenAI API has no choices.")
                raise CustomException("Response from OpenAI API has no choices", sys)

            message_content = response.choices[0].message.content
            if message_content is None:
                logger.error("Choice message content is None.")
                raise CustomException("Response from OpenAI API has no content", sys)

            if not str(message_content).strip():
                logger.warning("Response content is an empty string.")
                raise CustomException("Response from OpenAI API contains empty text", sys)

            usage = response.usage
            usage_data = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }

            latency = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Generation complete. Latency: {latency:.2f}ms. Total Tokens: {usage_data['total_tokens']}")

            return LLMResponse(
                content=message_content,
                raw_response=str(message_content),
                model_name=self.model_name,
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                latency_ms=round(latency, 2),
                provider=self.__class__.__name__,
                metadata={
                    "provider": "openai",
                    "model": self.model_name,
                    "request_id": getattr(response, "id", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                },
            )
        except RateLimitError as exc:
            logger.warning(f"Rate limit error encountered, retrying: {exc}")
            raise
        except Exception as exc:
            logger.error(f"Error during OpenAI Chat Completion: {str(exc)}")
            logger.exception("Full traceback for API failure:")
            raise CustomException(exc, sys)
