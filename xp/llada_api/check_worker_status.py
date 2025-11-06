#!/usr/bin/env python3
"""
Quick worker status checker for debugging load balancer issues.
"""

import asyncio
import argparse
import time
import sys
import json
from typing import Dict, Any

import httpx


async def check_worker_status(load_balancer_url: str = "http://localhost:8000", verbose: bool = False):
    """Check the status of all workers via the load balancer."""
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            print(f"🔍 Checking worker status at {load_balancer_url}")
            print(f"{'='*60}")
            
            # Get worker status
            response = await client.get(f"{load_balancer_url}/worker-status")
            if response.status_code != 200:
                print(f"❌ Error: Load balancer returned {response.status_code}")
                print(response.text)
                return False
            
            data = response.json()
            
            # Summary
            healthy_count = data["healthy_workers"]
            total_count = data["total_workers"]
            system_status = data.get("system_status", "unknown")
            
            print(f"📊 SYSTEM STATUS: {system_status.upper()}")
            print(f"🏥 HEALTHY WORKERS: {healthy_count}/{total_count}")
            print()
            
            # Worker details
            print(f"{'ID':<3} {'STATUS':<10} {'REQUESTS':<8} {'ERRORS':<6} {'LAST ACTIVE':<12} {'CONSECUTIVE ERRORS':<17}")
            print(f"{'-'*3} {'-'*10} {'-'*8} {'-'*6} {'-'*12} {'-'*17}")
            
            for worker in data["workers"]:
                worker_id = worker["worker_id"]
                status = worker["status"].upper()
                requests = worker["requests_served"]
                errors = worker["errors"]
                last_active = f"{worker['last_activity_seconds_ago']}s ago"
                consecutive_errors = worker["consecutive_errors"]
                
                # Color coding
                if status == "HEALTHY":
                    status_display = f"✅ {status}"
                elif status == "BUSY":
                    status_display = f"🔄 {status}"
                else:
                    status_display = f"❌ {status}"
                
                print(f"{worker_id:<3} {status_display:<10} {requests:<8} {errors:<6} {last_active:<12} {consecutive_errors:<17}")
            
            print()
            
            # Issues detection
            issues = []
            
            if healthy_count == 0:
                issues.append("🚨 NO HEALTHY WORKERS - All requests will fail!")
            elif healthy_count < total_count:
                issues.append(f"⚠️ {total_count - healthy_count} workers are unhealthy")
            
            busy_workers = [w for w in data["workers"] if w["status"] == "busy"]
            if len(busy_workers) == total_count:
                issues.append("📈 All workers are busy - system under high load")
            
            high_error_workers = [w for w in data["workers"] if w["consecutive_errors"] >= 2]
            if high_error_workers:
                issues.append(f"⚠️ {len(high_error_workers)} workers have consecutive errors")
            
            if issues:
                print("🔍 DETECTED ISSUES:")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print("✅ No issues detected")
            
            print()
            
            # Verbose output
            if verbose:
                print("📋 RAW DATA:")
                print(json.dumps(data, indent=2))
            
            return healthy_count > 0
            
        except httpx.RequestError as e:
            print(f"❌ Connection error: {e}")
            print("Make sure the load balancer is running and accessible")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False


async def monitor_workers(load_balancer_url: str, interval: int = 10):
    """Continuously monitor worker status."""
    print(f"🔄 Monitoring workers every {interval}s. Press Ctrl+C to stop.")
    print()
    
    try:
        while True:
            success = await check_worker_status(load_balancer_url)
            if not success:
                print("⚠️ Monitor check failed, retrying...")
            
            print(f"\n⏰ Next check in {interval}s...")
            await asyncio.sleep(interval)
            print("\033[2J\033[H")  # Clear screen
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Check worker status via load balancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick status check
  python check_worker_status.py
  
  # Check specific load balancer
  python check_worker_status.py --url http://remote-host:8000
  
  # Monitor continuously
  python check_worker_status.py --monitor --interval 5
  
  # Verbose output with raw data
  python check_worker_status.py --verbose
        """
    )
    
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Load balancer URL (default: http://localhost:8000)")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor continuously instead of single check")
    parser.add_argument("--interval", type=int, default=10,
                       help="Monitor interval in seconds (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose output with raw data")
    
    args = parser.parse_args()
    
    if args.monitor:
        asyncio.run(monitor_workers(args.url, args.interval))
    else:
        success = asyncio.run(check_worker_status(args.url, args.verbose))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
