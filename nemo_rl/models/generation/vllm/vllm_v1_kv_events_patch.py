# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Monkey patch for vLLM V1 Scheduler to add debug logging for KV cache events.

This patch adds logging to help debug why KV events might not be reaching
the ZMQ publisher, even though the code path exists.

Usage:
    Import this module after vLLM is initialized:
    
    from nemo_rl.models.generation.vllm.vllm_v1_kv_events_patch import patch_vllm_v1_kv_events
    patch_vllm_v1_kv_events()
"""

import logging

logger = logging.getLogger(__name__)

_patched = False


def patch_vllm_v1_kv_events():
    """
    Patch vLLM V1 Scheduler to add debug logging for KV cache event publishing.
    
    This helps diagnose why events might not be published even though the
    infrastructure exists.
    """
    global _patched
    
    if _patched:
        logger.info("vLLM V1 KV events debug patch already applied")
        return
    
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        
        logger.info("Applying vLLM V1 KV events debug patch...")
        
        # Save original schedule method
        original_schedule = Scheduler.schedule
        
        def patched_schedule(self):
            """Patched schedule with debug logging for KV events."""
            result = original_schedule(self)
            
            # Track schedule calls
            if not hasattr(patched_schedule, '_call_count'):
                patched_schedule._call_count = 0
                patched_schedule._events_published = 0
            patched_schedule._call_count += 1
            
            # Check if we collected any events
            if hasattr(self, 'kv_cache_manager'):
                events = self.kv_cache_manager.take_events()
                
                if events:
                    patched_schedule._events_published += len(events)
                    logger.info(
                        f"[KV PUBLISH] Worker published {len(events)} events "
                        f"(total: {patched_schedule._events_published}), "
                        f"publisher: {type(self.kv_event_publisher).__name__}"
                    )
                    
                    # Log first event details for initial events
                    if patched_schedule._events_published <= 10 and len(events) > 0:
                        first_event = events[0]
                        logger.info(
                            f"[KV PUBLISH] First event type: {type(first_event).__name__}"
                        )
                    
                    # Try to publish them (this mimics what schedule() should do)
                    try:
                        from vllm.distributed.kv_events import KVEventBatch
                        import time
                        
                        batch = KVEventBatch(ts=time.time(), events=events)
                        self.kv_event_publisher.publish(batch)
                    except Exception as e:
                        logger.error(f"[KV PUBLISH] Error publishing events: {e}", exc_info=True)
                else:
                    # Log every 1000 schedule calls if no events
                    if patched_schedule._call_count % 1000 == 1:
                        logger.debug(
                            f"[KV PUBLISH] No events after {patched_schedule._call_count} schedule calls, "
                            f"total events published: {patched_schedule._events_published}"
                        )
            
            return result
        
        # Apply the patch
        Scheduler.schedule = patched_schedule
        
        logger.info("✓ vLLM V1 KV events debug patch applied successfully")
        _patched = True
        
    except ImportError as e:
        logger.warning(f"Could not apply vLLM V1 KV events debug patch: {e}")
    except Exception as e:
        logger.error(f"Error applying vLLM V1 KV events debug patch: {e}", exc_info=True)


def is_patched() -> bool:
    """Check if the patch has been applied."""
    return _patched

