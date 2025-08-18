# SLURM Server Connection Output Pattern

## What the server now outputs when running on SLURM:

```
========== LLADA API SERVER STARTED - CONNECTION INFO ==========

----------[ 1. LOCAL TERMINAL: Create SSH Tunnel ]----------
Run this command on your LOCAL machine. It will seem to hang, which is normal.

   ssh -N -L 8000:gpu-node-001:8000 username@cluster.login.edu

   (Replace 'cluster.login.edu' with your cluster's login node address)
------------------------------------------------------------

----------[ 2. ACCESS THE API: Use Local URLs ]----------
After setting up the SSH tunnel, use these URLs on your LOCAL machine:

   API Base URL:    http://localhost:8000
   Health Check:    http://localhost:8000/health
   API Documentation: http://localhost:8000/docs

----------[ 3. TEST THE API ]----------
Test with curl:
   curl http://localhost:8000/health

Or use the Python client:
   python xp/llada_api/examples/llada_api_client.py
==============================================================
```

## Key Features Added:

1. **Automatic Connection Instructions**: Appears in job logs when server starts
2. **Compute Node Detection**: Uses `$(hostname)` to get the actual compute node
3. **Ready-to-Use SSH Commands**: Complete SSH tunnel command with node and port
4. **Local URL Access**: All URLs for local machine usage after tunnel setup
5. **Testing Commands**: Ready commands to verify the connection works

## Pattern Used:

The script monitors the server output and triggers when it sees:
```
"Uvicorn running on" + port number
```

This matches the FastAPI/Uvicorn startup message and immediately displays connection info.
