#!/bin/bash

# Helper script to connect to LLaDA server running on SLURM
# This script helps set up SSH tunnels to access the server running on a compute node

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

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_instruction() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script helps you connect to a LLaDA API server running on SLURM."
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -j, --job-id JOB_ID     SLURM job ID running the server"
    echo "  -n, --node NODE         Compute node name (if known)"
    echo "  -p, --port PORT         Server port (default: 8000)"
    echo "  -l, --login-node HOST   Login node hostname"
    echo ""
    echo "Examples:"
    echo "  # Find and connect to server automatically"
    echo "  $0 --job-id 12345"
    echo ""
    echo "  # Connect when you know the compute node"
    echo "  $0 --node gpu-001 --port 8000 --login-node cluster.example.com"
    echo ""
    echo "  # Just get connection instructions"
    echo "  $0 --help-connect"
}

show_connection_help() {
    cat <<EOF

${CYAN}=== How to Connect to LLaDA Server on SLURM ===${NC}

${GREEN}Step 1: Find your compute node and port${NC}
- Check your SLURM job logs for the server startup message
- Look for lines like: "Server will be accessible at: http://gpu-001:8000"

${GREEN}Step 2: Create SSH tunnel (run on your LOCAL machine)${NC}
SSH tunnel command (replace values as needed):
  ${YELLOW}ssh -N -L 8000:COMPUTE_NODE:8000 USER@LOGIN_NODE${NC}

Example:
  ${YELLOW}ssh -N -L 8000:gpu-001:8000 johndoe@cluster.example.com${NC}

This command will seem to hang - that's normal! Keep it running.

${GREEN}Step 3: Access the API${NC}
Once the SSH tunnel is running, you can access the server locally:
- ${CYAN}API Base URL:${NC} http://localhost:8000
- ${CYAN}Health Check:${NC} http://localhost:8000/health
- ${CYAN}API Documentation:${NC} http://localhost:8000/docs

${GREEN}Step 4: Test the connection${NC}
Test with curl:
  ${YELLOW}curl http://localhost:8000/health${NC}

Or use the Python client:
  ${YELLOW}python examples/llada_api_client.py${NC}

${GREEN}Step 5: When done${NC}
- Press Ctrl+C in the SSH tunnel terminal to close it
- Cancel your SLURM job: ${YELLOW}scancel JOB_ID${NC}

EOF
}

# Default values
JOB_ID=""
NODE=""
PORT="8000"
LOGIN_NODE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --help-connect)
            show_connection_help
            exit 0
            ;;
        -j|--job-id)
            JOB_ID="$2"
            shift 2
            ;;
        -n|--node)
            NODE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -l|--login-node)
            LOGIN_NODE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to get node from job ID
get_node_from_job() {
    local job_id=$1
    
    if ! command -v squeue &> /dev/null; then
        print_error "squeue command not found. Are you on a SLURM login node?"
        return 1
    fi
    
    # Get node list for the job
    local nodes=$(squeue -j "$job_id" -h -o "%N" 2>/dev/null)
    
    if [[ -z "$nodes" ]]; then
        print_error "Job $job_id not found or not running"
        return 1
    fi
    
    # Extract first node (assuming single-node job for LLaDA server)
    local first_node=$(echo "$nodes" | cut -d',' -f1 | sed 's/\[.*$//')
    echo "$first_node"
}

# Function to get job logs path
get_job_logs() {
    local job_id=$1
    
    # Try to find log file
    local possible_paths=(
        "logs/llada_server/llada_server_${job_id}.log"
        "${LOG}/llada_server/llada_server_${job_id}.log"
        "llada_server_${job_id}.log"
    )
    
    for path in "${possible_paths[@]}"; do
        if [[ -f "$path" ]]; then
            echo "$path"
            return 0
        fi
    done
    
    return 1
}

# Function to check if server is accessible via SSH tunnel
test_server_connection() {
    local local_port=$1
    
    print_info "Testing connection to http://localhost:$local_port/health"
    
    if command -v curl &> /dev/null; then
        if curl -s "http://localhost:$local_port/health" >/dev/null 2>&1; then
            print_info "✅ Server is accessible!"
            curl -s "http://localhost:$local_port/health" | python3 -m json.tool 2>/dev/null || echo "Health check succeeded"
        else
            print_warning "❌ Server not accessible. Check your SSH tunnel and server status."
        fi
    else
        print_warning "curl not found. Please test manually: http://localhost:$local_port/health"
    fi
}

# Main logic
main() {
    print_info "LLaDA SLURM Server Connection Helper"
    echo "=================================="
    
    # If job ID provided, try to get node automatically
    if [[ -n "$JOB_ID" ]]; then
        print_info "Looking up compute node for job $JOB_ID..."
        
        if NODE_FROM_JOB=$(get_node_from_job "$JOB_ID"); then
            NODE="$NODE_FROM_JOB"
            print_info "Found compute node: $NODE"
            
            # Try to find and show relevant log excerpts
            if LOG_PATH=$(get_job_logs "$JOB_ID"); then
                print_info "Job log file: $LOG_PATH"
                
                # Look for server startup messages
                if grep -q "Server will be accessible" "$LOG_PATH" 2>/dev/null; then
                    echo ""
                    print_info "Server status from logs:"
                    grep -A2 -B2 "Server will be accessible\|Health check\|API docs" "$LOG_PATH" | tail -10
                fi
            fi
        else
            print_error "Could not determine compute node for job $JOB_ID"
            exit 1
        fi
    fi
    
    # Validate we have the required information
    if [[ -z "$NODE" ]]; then
        print_error "Compute node not specified. Use --node or --job-id"
        show_usage
        exit 1
    fi
    
    # Try to determine login node if not provided
    if [[ -z "$LOGIN_NODE" ]]; then
        if [[ -n "$SLURM_SUBMIT_HOST" ]]; then
            LOGIN_NODE="$SLURM_SUBMIT_HOST"
            print_info "Using login node from SLURM environment: $LOGIN_NODE"
        else
            LOGIN_NODE="<YOUR_LOGIN_NODE>"
            print_warning "Login node not specified. You'll need to replace <YOUR_LOGIN_NODE> in the commands below."
        fi
    fi
    
    # Show connection instructions
    echo ""
    print_instruction "Connection Instructions:"
    echo ""
    
    print_info "1. Create SSH tunnel (run on your LOCAL machine):"
    echo "   ${YELLOW}ssh -N -L $PORT:$NODE:$PORT $USER@$LOGIN_NODE${NC}"
    echo ""
    
    print_info "2. Test the connection:"
    echo "   ${YELLOW}curl http://localhost:$PORT/health${NC}"
    echo ""
    
    print_info "3. Use the API:"
    echo "   • Base URL: ${CYAN}http://localhost:$PORT${NC}"
    echo "   • Health: ${CYAN}http://localhost:$PORT/health${NC}" 
    echo "   • Docs: ${CYAN}http://localhost:$PORT/docs${NC}"
    echo ""
    
    print_info "4. Test with Python client:"
    echo "   ${YELLOW}python xp/llada_api/examples/llada_api_client.py${NC}"
    echo ""
    
    print_info "💡 The job logs also show connection instructions when the server starts:"
    if [[ -n "$JOB_ID" ]]; then
        echo "   ${YELLOW}tail -f logs/llada_server/llada_server_${JOB_ID}.log${NC}"
    else
        echo "   ${YELLOW}tail -f logs/llada_server/llada_server_JOBID.log${NC}"
    fi
    echo ""
    
    # Offer to wait and test
    if [[ "$LOGIN_NODE" != "<YOUR_LOGIN_NODE>" ]]; then
        echo "Press Enter after setting up the SSH tunnel to test the connection, or Ctrl+C to exit..."
        read -r
        test_server_connection "$PORT"
    fi
    
    echo ""
    print_info "When finished, press Ctrl+C in the SSH tunnel terminal and cancel the SLURM job:"
    echo "   ${YELLOW}scancel $JOB_ID${NC}"
}

# Handle special case for help
if [[ $# -eq 0 ]]; then
    show_usage
    echo ""
    show_connection_help
    exit 0
fi

main
