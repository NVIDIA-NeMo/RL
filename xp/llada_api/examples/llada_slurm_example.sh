#!/bin/bash

# Example: End-to-end LLaDA API server on SLURM
# This script demonstrates how to:
# 1. Start a LLaDA server on SLURM with DCP checkpoint
# 2. Set up connection to the server
# 3. Test the API
# 4. Clean up

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration - MODIFY THESE FOR YOUR SETUP
DCP_CHECKPOINT_PATH="/path/to/your/checkpoint.dcp"  # Change this!
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
SERVER_PORT="8000"
JOB_TIME="2:00:00"
JOB_PARTITION="interactive"

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check ACCOUNT is set
    if [[ -z "$ACCOUNT" ]]; then
        print_error "ACCOUNT environment variable not set!"
        echo "Please set it: export ACCOUNT=your_slurm_account"
        exit 1
    fi
    print_info "SLURM account: $ACCOUNT"
    
    # Check DCP checkpoint path
    if [[ "$DCP_CHECKPOINT_PATH" == "/path/to/your/checkpoint.dcp" ]]; then
        print_error "Please modify DCP_CHECKPOINT_PATH in this script to point to your actual checkpoint!"
        exit 1
    fi
    
    # Check if scripts exist
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    START_SCRIPT="$SCRIPT_DIR/../scripts/start_llada_server.sh"
    CONNECT_SCRIPT="$SCRIPT_DIR/../scripts/connect_to_llada_server.sh"
    
    if [[ ! -f "$START_SCRIPT" ]]; then
        print_error "start_llada_server.sh not found at $START_SCRIPT"
        exit 1
    fi
    
    if [[ ! -f "$CONNECT_SCRIPT" ]]; then
        print_error "connect_to_llada_server.sh not found at $CONNECT_SCRIPT"
        exit 1
    fi
    
    print_info "Prerequisites check passed!"
}

# Step 1: Start the server
start_server() {
    print_step "Starting LLaDA API server on SLURM..."
    
    # Submit the job in background
    "$START_SCRIPT" \
        --dcp-path "$DCP_CHECKPOINT_PATH" \
        --base-model "$BASE_MODEL" \
        --port "$SERVER_PORT" \
        --time "$JOB_TIME" \
        --partition "$JOB_PARTITION" \
        --job-name "llada-demo-server" &
    
    SERVER_PID=$!
    
    print_info "Server submission started (PID: $SERVER_PID)"
    print_info "Waiting for job to be allocated..."
    
    # Give it some time to start
    sleep 10
    
    # Find the job ID
    JOB_ID=$(squeue -u $USER --name=llada-demo-server -h -o "%i" | head -1)
    
    if [[ -z "$JOB_ID" ]]; then
        print_error "Could not find submitted job. Check squeue -u $USER"
        exit 1
    fi
    
    export JOB_ID
    print_info "Job ID: $JOB_ID"
    
    # Wait for job to be running
    print_info "Waiting for job to start running..."
    while [[ $(squeue -j $JOB_ID -h -o "%T") != "RUNNING" ]]; do
        echo -n "."
        sleep 5
    done
    echo ""
    
    print_info "Job is now running!"
    
    # Show how to access logs for connection info
    echo ""
    print_info "📋 The server will display connection instructions when it starts."
    print_info "Watch the job logs:"
    echo "   tail -f logs/llada_server/llada_server_$JOB_ID.log"
    echo ""
}

# Step 2: Get connection info
get_connection_info() {
    print_step "Getting connection information..."
    
    print_info "The server automatically displays connection instructions when it starts."
    print_info "You should see SSH tunnel commands in the job logs."
    echo ""
    
    # Also use the connection helper as backup
    print_info "Here are the manual connection instructions:"
    "$CONNECT_SCRIPT" --job-id $JOB_ID
}

# Step 3: Wait for user to set up tunnel
wait_for_tunnel() {
    print_step "SSH Tunnel Setup"
    
    echo ""
    echo "=================================================="
    echo "NEXT STEPS:"
    echo "1. Check the job logs for server startup and connection info:"
    echo "   tail -f logs/llada_server/llada_server_$JOB_ID.log"
    echo ""
    echo "2. Open a NEW terminal on your LOCAL machine"
    echo "3. Run the SSH tunnel command from the logs (or above)"
    echo "4. Come back here and press Enter to continue testing"
    echo "=================================================="
    echo ""
    
    read -p "Press Enter after setting up the SSH tunnel..."
}

# Step 4: Test the API
test_api() {
    print_step "Testing the API..."
    
    # Test health endpoint
    print_info "Testing health endpoint..."
    if curl -s "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
        print_info "✅ Health check passed!"
        curl -s "http://localhost:$SERVER_PORT/health" | python3 -m json.tool
    else
        print_error "❌ Health check failed. Is your SSH tunnel working?"
        return 1
    fi
    
    echo ""
    
    # Test chat completion
    print_info "Testing chat completion..."
    
    local test_request=$(cat <<EOF
{
    "messages": [{"role": "user", "content": "What is 2+2? Give a very brief answer."}],
    "temperature": 0.0,
    "max_tokens": 50,
    "steps": 16
}
EOF
)
    
    print_info "Sending test request..."
    
    local response=$(curl -s -X POST "http://localhost:$SERVER_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$test_request")
    
    if echo "$response" | python3 -c "import sys, json; json.load(sys.stdin)" >/dev/null 2>&1; then
        print_info "✅ Chat completion test passed!"
        
        # Extract and display the response
        local content=$(echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'choices' in data and len(data['choices']) > 0:
    print(data['choices'][0]['message']['content'])
else:
    print('No content in response')
")
        
        echo "Response: \"$content\""
        echo ""
        
        # Show usage stats
        local usage=$(echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'usage' in data:
    usage = data['usage']
    print(f\"Tokens - Input: {usage.get('prompt_tokens', 'N/A')}, Output: {usage.get('completion_tokens', 'N/A')}, Total: {usage.get('total_tokens', 'N/A')}\")
")
        
        print_info "Usage: $usage"
        
    else
        print_error "❌ Chat completion test failed!"
        echo "Response: $response"
        return 1
    fi
}

# Step 5: Run Python client example
test_python_client() {
    print_step "Testing with Python client..."
    
    CLIENT_SCRIPT="$SCRIPT_DIR/llada_api_client.py"
    if [[ -f "$CLIENT_SCRIPT" ]]; then
        print_info "Running Python client example..."
        python3 "$CLIENT_SCRIPT" || print_warning "Python client test had issues (this might be normal)"
    else
        print_warning "Python client example not found, skipping..."
    fi
}

# Step 6: Cleanup
cleanup() {
    print_step "Cleaning up..."
    
    if [[ -n "$JOB_ID" ]]; then
        print_info "Canceling SLURM job $JOB_ID..."
        scancel $JOB_ID
        print_info "Job canceled."
    fi
    
    if [[ -n "$SERVER_PID" ]]; then
        print_info "Stopping server submission process..."
        kill $SERVER_PID 2>/dev/null || true
    fi
    
    print_info "Cleanup completed."
    print_info "Don't forget to close your SSH tunnel (Ctrl+C in that terminal)!"
}

# Main execution
main() {
    echo "=================================================="
    echo "🚀 LLaDA SLURM API Server Demo"
    echo "=================================================="
    echo ""
    
    # Trap to cleanup on exit
    trap cleanup EXIT
    
    check_prerequisites
    start_server
    get_connection_info
    wait_for_tunnel
    
    if test_api; then
        test_python_client
        
        echo ""
        print_info "🎉 Demo completed successfully!"
        echo ""
        print_info "Your LLaDA server is running and accessible at:"
        print_info "  http://localhost:$SERVER_PORT"
        echo ""
        print_info "You can now:"
        print_info "  • Use the Python client: python examples/llada_api_client.py"
        print_info "  • Access the docs: http://localhost:$SERVER_PORT/docs"
        print_info "  • Make API calls to: http://localhost:$SERVER_PORT/v1/chat/completions"
        echo ""
        
        read -p "Press Enter when you're done experimenting to clean up..."
        
    else
        print_error "❌ Demo failed during API testing."
        print_info "Check the job logs for more details:"
        print_info "  tail logs/llada_server/llada_server_$JOB_ID.log"
    fi
}

# Check if running directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
