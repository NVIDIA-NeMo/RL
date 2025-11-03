import asyncio
import random
import math
from sys import int_info
from typing import AsyncIterator, Dict, List, Tuple, Optional
from collections import deque

from dynamo._core import Context, DistributedRuntime, KvPushRouter, KvRouterConfig


class Router:
    def __init__(self, runtime: DistributedRuntime, namespace: str, component: str, endpoint: str, served_model_name: str):
        self.runtime = runtime
        self.namespace = self.runtime.namespace(namespace)
        self.component = self.namespace.component(component)
        self.endpoint = self.component.endpoint(endpoint)
        self.served_model_name = served_model_name
    
    async def initialize(self):
        pass

    async def route_request(self, request: Dict) -> Tuple[AsyncIterator[Dict], Context]:
        """Base route_request method - override in subclasses.
        
        Returns:
            Tuple of (response iterator, context for cancellation)
        """
        raise NotImplementedError("Subclasses must implement route_request()")

class KvRouter(Router):
    def __init__(self, runtime: DistributedRuntime, namespace: str, component: str, endpoint: str, served_model_name: str):
        super().__init__(runtime, namespace, component, endpoint, served_model_name)

    async def initialize(self, block_size: int):
        self.block_size = block_size

        kv_router_config = KvRouterConfig()   
        self.kv_push_router = KvPushRouter(
            endpoint=self.endpoint,
            block_size=self.block_size,
            kv_router_config=kv_router_config,
        )

        self.client = await self.endpoint.client()

        await self.client.wait_for_instances()
    
    async def route_request(self, request: Dict) -> Tuple[AsyncIterator[Dict], Context]:
        """Route request using KV-aware routing.
        
        Returns:
            Tuple of (response iterator, context object)
        """
        # TODO: (sechoi) the following only works for mocker
        token_ids = [random.randint(1, 100) for _ in range(10)]
        
        # Create a context for this request
        context = Context()

        # Prepare sampling options
        # temperature = 0.0 if greedy else self.cfg.get("temperature", 0.7)
        sampling_options = {
            "temperature": 0.0,
        }
        
        # Prepare stop conditions (max_tokens must be here for mocker)
        # Use random value between 1 and allowed_new_tokens for variable generation lengths
        random_max_tokens = random.randint(1, max(1, 4096))
        stop_conditions = {
            "max_tokens": random_max_tokens,  # Random for variable lengths
        }

        response_iterator = await self.kv_push_router.generate(
            token_ids=token_ids,
            model=self.served_model_name,
            stop_conditions=stop_conditions,
            sampling_options=sampling_options,
        )

        return response_iterator, context

class RoundRobinRouter(Router):
    def __init__(self, runtime: DistributedRuntime, namespace: str, component: str, endpoint: str, served_model_name: str):
        super().__init__(runtime, namespace, component, endpoint, served_model_name)

    async def initialize(self, block_size: int, max_model_len: int):
        self.max_model_len = max_model_len
        self.client = await self.endpoint.client()
        await self.client.wait_for_instances()

        self.instance_ids = self.client.instance_ids()
        if not self.instance_ids:
            raise RuntimeError("No instances available")
        self.instance_id_index = 0

    async def route_request(self, request: Dict) -> Tuple[AsyncIterator[Dict], Context]:
        """Route request with proper OpenAI chat completion format.
        
        Args:
            request: Dict with keys like 'messages', 'temperature', 'max_tokens', etc.
        
        Returns:
            Tuple of (AsyncIterator of response chunks, Context object for cancellation)
        """

        meta_data = {}

        token_ids = request.get("token_ids", [random.randint(1, 100) for _ in range(10)])
        # LogNormal distribution skewed toward higher values
        # LogNormal(μ=7.0, σ=1.1)
        # mu = 7.0
        # sigma = 1.1


        # random_max_tokens = int(min(request["remaining_ctx"], math.exp(random.gauss(mu, sigma))))
        random_max_tokens = int(request["remaining_ctx"])
        # random_max_tokens = request.get("remaining_ctx", math.exp(random.gauss(mu, sigma)))

        

        # Select instance using round-robin
        selected_instance = self.instance_ids[self.instance_id_index % len(self.instance_ids)]
        self.instance_id_index += 1

        meta_data["max_tokens"] = random_max_tokens
        meta_data["selected_instance"] = selected_instance

        formatted_request = {
            "token_ids": token_ids,
            "model": request.get("model", self.served_model_name),
            "stop_conditions": {
                "max_tokens": random_max_tokens
            },
            "sampling_options": {
                "temperature": 0.7,
            },
            "output_options": {
                "include_input_tokens": False,
                "return_full_text": False,
            },
            "eos_token_ids": [],
            "annotations": [],  # Required field
        }
        
        context = Context()
        # response_iterator = await self.client.round_robin(formatted_request, context=context)
        response_iterator = await self.client.direct(formatted_request, selected_instance, context=context)
        
        return response_iterator, context, meta_data

# class APRILRouter(Router):
#     """
#     Active Partial Rollouts in Reinforcement Learning (APRIL) Router.
    
#     Based on the paper: https://arxiv.org/abs/2509.18521
    
#     APRIL addresses the long-tail distribution problem in RL rollout generation by:
#     1. Over-provisioning rollout requests (sending more than needed)
#     2. Terminating once the target number of complete responses is reached
#     3. Recycling incomplete responses for continuation in future steps
    
#     This reduces GPU idle time caused by waiting for slow, lengthy responses.
#     """
    
#     def __init__(
#         self, 
#         runtime: DistributedRuntime, 
#         namespace: str, 
#         component: str, 
#         endpoint: str, 
#         served_model_name: str,
#         over_sampling_batch_size: int_info,
#         rollout_batch_size: int
#     ):
#         """
#         Args:
#             runtime: Dynamo distributed runtime
#             namespace: Namespace for the endpoint
#             component: Component name
#             endpoint: Endpoint name
#             served_model_name: Model identifier
#             over_sampling_batch_size: Batch size of over-provision requests
#             rollout_batch_size: Batch size of rollout requests
#         """
#         super().__init__(runtime, namespace, component, endpoint, served_model_name)
#         self.over_sampling_batch_size = over_sampling_batch_size
#         self.rollout_batch_size = rollout_batch_size

#         assert self.over_sampling_batch_size >= self.rollout_batch_size, "Over-sampling batch size must be greater than or equal to rollout batch size"
        
#         self.partial_responses_queue = deque()  # Queue for incomplete responses to recycle

#     async def initialize(self, block_size: int):
#         self.client = await self.endpoint.client()
#         await self.client.wait_for_instances()

#     def _prepare_request(self, request: Dict, is_continuation: bool = False) -> Dict:
#         """
#         Prepare a request in the PreprocessedRequest format.
        
#         Args:
#             request: Input request dict
#             is_continuation: Whether this is a continuation of a partial response
#         """
#         # Extract or generate token_ids
#         if "token_ids" in request:
#             token_ids = request["token_ids"]
#         else:
#             # For testing: generate random tokens
#             token_ids = [random.randint(1, 100) for _ in range(10)]
        
#         max_tokens = request.get("max_tokens", random.randint(10, 100))
        
#         formatted_request = {
#             "token_ids": token_ids,
#             "model": request.get("model", self.served_model_name),
#             "stop_conditions": {
#                 "max_tokens": max_tokens
#             },
#             "sampling_options": {
#                 "temperature": request.get("temperature", 0.7),
#             },
#             "output_options": request.get("output_options", {}),
#             "eos_token_ids": request.get("eos_token_ids", []),
#             "annotations": request.get("annotations", []),
#         }
        
#         return formatted_request

#     async def route_request(self, request: Dict) -> AsyncIterator[Dict]:
#         """
#         Route a single request (for compatibility with base Router interface).
#         """
#         # Use provided context or create a new one
#         context = Context()

#         request = self._prepare_request(request)

#         response_stream = await self.client.round_robin(request, context=context)

#         async for response in response_stream:
#             # Extract data from Annotated wrapper
#             response_data = response.data() if hasattr(response, 'data') else response
#             if response_data:
#                 yield response_data
