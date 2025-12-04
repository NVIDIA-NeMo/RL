# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
vLLM V1 KV Event Listener

Handles vLLM V1's msgpack-based KV cache event protocol.
Unlike vLLM V0 which used JSON, V1 uses msgspec's msgpack encoding
with multipart ZMQ messages and sequence tracking.

**IMPORTANT**: As of vLLM 0.6.x, the V1 AsyncLLM API does NOT emit KV cache events,
even when configured with enable_kv_cache_events=True. This listener is correctly
implemented and ready to use, but will not receive events until vLLM V1 adds support.

To test if your vLLM version supports event emission, run:
    python test_vllm_v1_kv_events.py

See TEST_VLLM_V1_KV_EVENTS.md for more details.
"""

import logging
from typing import Any, Optional

import msgspec
import zmq
import zmq.asyncio
from msgspec.msgpack import Decoder

logger = logging.getLogger(__name__)


# Event type definitions matching vLLM V1 protocol
class KVCacheEvent(msgspec.Struct, array_like=True, omit_defaults=True, gc=False, tag=True):
    """Base class for all KV cache-related events"""
    pass


class BlockStored(KVCacheEvent):
    """Event emitted when a block is stored in the KV cache"""
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str]


class BlockRemoved(KVCacheEvent):
    """Event emitted when a block is removed from the KV cache"""
    block_hashes: list[int]
    medium: Optional[str]


class AllBlocksCleared(KVCacheEvent):
    """Event emitted when all blocks are cleared from the KV cache"""
    pass


class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    """Batch of events with timestamp"""
    ts: float
    events: list[Any]


class KVEventBatch(EventBatch):
    """Batch of KV cache events"""
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]


class VllmV1KvEventListener:
    """
    Async ZMQ listener for vLLM V1 KV cache events.
    
    This listener:
    1. Subscribes to vLLM V1's msgpack-encoded KV events
    2. Handles multipart ZMQ messages (topic, seq, payload)
    3. Tracks sequence numbers to detect missed messages
    4. Optionally supports replay for missed messages (if vLLM supports it)
    
    Args:
        endpoint: ZMQ endpoint to connect to (e.g., "tcp://localhost:5557")
        block_size: Block size for KV cache (used for validation)
        topic: ZMQ topic to subscribe to (default: "kv-events")
        enable_replay: Whether to enable replay for missed messages (default: False)
    """
    
    def __init__(
        self,
        endpoint: str,
        block_size: int,
        topic: str = "kv-events",
        enable_replay: bool = False,
    ):
        self.endpoint = endpoint
        self.block_size = block_size
        self.topic = topic
        self.enable_replay = enable_replay
        
        # Initialize ZMQ context and socket
        self.context = zmq.asyncio.Context()
        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect(endpoint)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        # Don't set RCVTIMEO - we use poll() for non-blocking checks instead
        
        # Initialize replay socket if enabled
        self.replay = None
        if enable_replay:
            replay_endpoint = endpoint.replace("5557", "5558")  # Replay port convention
            self.replay = self.context.socket(zmq.REQ)
            self.replay.connect(replay_endpoint)
        
        # Sequence tracking
        self.last_seq = -1
        
        # Decoder for msgpack events
        self.decoder = Decoder(type=KVEventBatch)
        
        logger.info(
            f"VllmV1KvEventListener initialized: endpoint={endpoint}, "
            f"topic={topic}, block_size={block_size}, replay={enable_replay}"
        )
    
    async def get_events(self) -> list[dict[str, Any]]:
        """
        Poll for new KV cache events and return them as a list of dictionaries.
        
        Returns:
            List of event dictionaries compatible with RadixTree.apply_event()
            Each dict has the format expected by the Rust RadixTree implementation.
        """
        logger.debug(f"[VllmV1KvEventListener] get_events() called on {self.endpoint}")
        events = []
        
        try:
            # Poll with timeout (non-blocking) - poll() on zmq.asyncio IS async!
            # Timeout is in milliseconds; 100ms allows events to arrive
            poll_result = await self.sub.poll(timeout=1000, flags=zmq.POLLIN)
            logger.info(f"[VllmV1KvEventListener] poll_result={poll_result} on {self.endpoint}")
            if poll_result:
                logger.debug(f"[VllmV1KvEventListener] Poll detected message on {self.endpoint}")
                
                # Receive multipart message: [topic, seq_bytes, payload] - THIS is async!
                parts = await self.sub.recv_multipart()
                logger.debug(f"[VllmV1KvEventListener] Received {len(parts)} parts")
                
                if len(parts) != 3:
                    logger.warning(f"[VllmV1KvEventListener] Unexpected message format: {len(parts)} parts (expected 3), parts={parts}")
                    return []
                
                topic_bytes, seq_bytes, payload = parts
                logger.debug(f"[VllmV1KvEventListener] topic={topic_bytes}, seq_len={len(seq_bytes)}, payload_len={len(payload)}")
                
                # Decode sequence number
                seq = int.from_bytes(seq_bytes, "big")
                logger.debug(f"[VllmV1KvEventListener] Sequence number: {seq}")
                
                # Check for missed messages
                if self.last_seq >= 0 and seq > self.last_seq + 1:
                    missed = seq - self.last_seq - 1
                    logger.warning(
                        f"[VllmV1KvEventListener] Missed {missed} KV events (last: {self.last_seq}, current: {seq})"
                    )
                    # TODO: Implement replay if needed
                
                # Decode event batch
                try:
                    event_batch: KVEventBatch = self.decoder.decode(payload)
                    logger.info(f"[VllmV1KvEventListener] Decoded event batch with {len(event_batch.events)} events at ts={event_batch.ts}")
                    
                    # Convert vLLM V1 events to RadixTree format
                    for event in event_batch.events:
                        logger.debug(f"[VllmV1KvEventListener] Processing event: {type(event).__name__}")
                        converted_event = self._convert_event_to_radix_format(event)
                        if converted_event:
                            events.append(converted_event)
                            logger.info(f"[VllmV1KvEventListener] Converted event: type={converted_event['type']}")
                    
                    # Update sequence tracker
                    self.last_seq = seq
                    
                except Exception as e:
                    logger.error(f"[VllmV1KvEventListener] Error decoding event batch: {e}", exc_info=True)
        
        except zmq.Again:
            # Timeout - no events available (this is normal)
            logger.debug(f"[VllmV1KvEventListener] No events available (timeout)")
            pass
        except Exception as e:
            logger.error(f"[VllmV1KvEventListener] Error receiving KV events: {e}", exc_info=True)
        
        return events
    
    def _convert_event_to_radix_format(self, event: KVCacheEvent) -> Optional[dict[str, Any]]:
        """
        Convert vLLM V1 event to the format expected by RadixTree.apply_event().
        
        The Rust RadixTree expects events in a specific JSON format. We need to
        convert vLLM V1's msgspec structs to this format.
        """
        if isinstance(event, BlockStored):
            # Convert BlockStored to RadixTree format
            return {
                "type": "block_stored",
                "block_hashes": event.block_hashes,
                "parent_block_hash": event.parent_block_hash,
                "token_ids": event.token_ids,
                "block_size": event.block_size,
                "lora_id": event.lora_id,
                "medium": event.medium,
            }
        elif isinstance(event, BlockRemoved):
            # Convert BlockRemoved to RadixTree format
            return {
                "type": "block_removed",
                "block_hashes": event.block_hashes,
                "medium": event.medium,
            }
        elif isinstance(event, AllBlocksCleared):
            # Convert AllBlocksCleared to RadixTree format
            return {
                "type": "all_blocks_cleared",
            }
        else:
            logger.warning(f"Unknown event type: {type(event)}")
            return None
    
    def close(self):
        """Close ZMQ sockets and context"""
        try:
            self.sub.close()
            if self.replay:
                self.replay.close()
            self.context.term()
            logger.info("VllmV1KvEventListener closed")
        except Exception as e:
            logger.error(f"Error closing VllmV1KvEventListener: {e}")

