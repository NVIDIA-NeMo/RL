"""Validate online W&B credentials without exposing or persisting credentials."""
import argparse,json,os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if os.environ.get('WANDB_MODE')!='online' or not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B mode and exported WANDB_API_KEY are required')
    import wandb
    entity=wandb.Api(timeout=45).default_entity
    if not entity:raise RuntimeError('W&B authentication did not resolve an entity')
    record={'status':'pass','entity':entity,'mode':'online','source_artifact_uploads':False}
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
