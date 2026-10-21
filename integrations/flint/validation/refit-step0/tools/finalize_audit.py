# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correct audit-only expert labels; preserve raw receipts and tensor payloads.

The running probe omitted the outer TRTLLM w1/w3 swap. This script checks
complete hashes against the opposite raw slice, then writes correctly named
safetensors without modifying any model or any captured numerical value.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct


def canonical_key(name):
    if re.fullmatch(r'.*\.mlp\.experts\.\d+\.(gate_proj|up_proj)\.weight',name):
        if '.gate_proj.' in name: return name.replace('.gate_proj.','.up_proj.')
        return name.replace('.up_proj.','.gate_proj.')
    return name


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def rewrite_shard(source,destination):
    with source.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(length))
        renamed={canonical_key(k):v for k,v in header.items()}
        assert len(renamed)==len(header)
        raw=json.dumps(renamed,separators=(',',':')).encode()
        raw+=b' '*((-len(raw))%8)
        before=hashlib.sha256()
        with destination.open('wb') as out:
            out.write(struct.pack('<Q',len(raw)));out.write(raw)
            for block in iter(lambda:f.read(8*1024*1024),b''):
                before.update(block);out.write(block)
    after=hashlib.sha256()
    with destination.open('rb') as f:
        f.seek(8+len(raw))
        for block in iter(lambda:f.read(8*1024*1024),b''):after.update(block)
    assert before.digest()==after.digest()
    return {'file':destination.name,'payload_sha256':after.hexdigest(),'payload_unchanged':True}


def corrected_comparison(path,out):
    original=json.loads(path.read_text())
    by_name={r['name']:r for r in original['records']}
    records=[]
    for name,row in by_name.items():
        key=canonical_key(name)
        fixed=copy.deepcopy(row)
        if key!=name:
            actual=by_name[key]
            for field in ['actual_dtype','actual_shape','actual_sha256','actual_sample']:
                fixed[field]=actual[field]
            exact=(fixed['actual_sha256']==fixed['original_sha256'] and fixed['actual_dtype']==fixed['original_dtype'] and fixed['actual_shape']==fixed['original_shape'] and actual['nonfinite_elements']==0)
            fixed.update(value_equal=exact,shape_match=fixed['actual_shape']==fixed['original_shape'],
                         differing_elements=0 if exact else None,max_abs_error=0.0 if exact else None,
                         mean_abs_error=0.0 if exact else None,nonfinite_elements=actual['nonfinite_elements'],
                         original_probe_stored_key=key,verification='Complete tensor SHA256 equality after documented gate/up permutation')
        records.append(fixed)
    report={**original,'records':records,'status':'pass' if all(r['value_equal'] for r in records) else 'mismatch',
        'unmapped_parameter_disposition':'Checked separately against documented zero defaults in reviewed-step0-comparison.json',
        'correction':'The original probe omitted _weight_loader_impl FlashInfer TRTLLM w1/w3 swap; names are corrected, numerical payloads preserved.'}
    out.write_text(json.dumps(report,indent=2)+'\n')
    return report


def finalize(root):
    completed=[]
    for folder in sorted((root/'tensors').iterdir()):
        raw_path=folder/'step0-comparison.json'
        if not raw_path.exists():continue
        reviewed=folder/'reviewed-step0-comparison.json'
        if reviewed.exists():
            completed.append(json.loads(reviewed.read_text()));continue
        raw=json.loads(raw_path.read_text())
        initial=corrected_comparison(folder/'initial-logical/comparison.json',folder/'initial-logical/corrected-comparison.json')
        logical=corrected_comparison(folder/'step0-logical/comparison.json',folder/'step0-logical/corrected-comparison.json')
        incoming=[json.loads(s) for s in (folder/'incoming-refit.jsonl').read_text().splitlines()]
        source_names={r['name'] for r in logical['records']}
        expected_defaults={f'model.layers.{i}.mlp.gate.expert_bias' for i in [2,3]}
        defaults=[]
        for p in raw['resident_records']:
            if p['name'] in expected_defaults:
                zero=(p['actual_dtype']=='torch.float32' and p['actual_shape']==[256] and
                      p['actual_sha256']==hashlib.sha256(bytes(256*4)).hexdigest())
                defaults.append({'name':p['name'],'all_values_zero':zero,'actual_sha256':p['actual_sha256']})
        default_pass=(set(initial['unmapped_resident_parameters'])==expected_defaults and
                      set(logical['unmapped_resident_parameters'])==expected_defaults and
                      len(defaults)==2 and all(r['all_values_zero'] for r in defaults))
        source_dir=folder/'step0-logical/tensors'
        target=folder/'step0-canonical'
        target.mkdir(exist_ok=True)
        manifest=[]
        index=json.loads((source_dir/'model.safetensors.index.json').read_text())['weight_map']
        for name in sorted(set(index.values())):
            manifest.append(rewrite_shard(source_dir/name,target/name))
        canonical_index={canonical_key(k):v for k,v in index.items()}
        assert set(canonical_index)==source_names
        (target/'model.safetensors.index.json').write_text(json.dumps({'weight_map':canonical_index},indent=2)+'\n')
        (target/'payload-verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
        report={'replica':folder.name,'status':'pass' if (initial['status']=='pass' and logical['status']=='pass'
            and raw['incoming_values_equal'] and raw['incoming_source_coverage'] and raw['incoming_count']==1589
            and raw['incoming_unique_count']==1589 and raw['postprocessed_resident_values_equal_initial']
            and raw['forwards_during_refit']==0 and default_pass) else 'mismatch',
            'initial_checkpoint_values_equal':initial['status']=='pass','step0_checkpoint_values_equal':logical['status']=='pass',
            'incoming_values_equal':raw['incoming_values_equal'],'incoming_count':raw['incoming_count'],
            'resident_parameter_count':raw['resident_parameter_count'],
            'postprocessed_resident_values_equal_initial':raw['postprocessed_resident_values_equal_initial'],
            'timestamp_utc':raw['timestamp_utc'],'forwards_during_refit':raw['forwards_during_refit'],
            'default_router_biases':defaults,'default_router_biases_pass':default_pass,
            'raw_receipt_retained':'step0-comparison.json','canonical_dump':'step0-canonical',
            'correction':'Audit-only gate/up labeling correction and explicit verification of two zero default biases. No runtime weights changed.'}
        reviewed.write_text(json.dumps(report,indent=2)+'\n');completed.append(report)
        print(folder.name,report['status'],flush=True)
    if len(completed)!=4:raise RuntimeError(f'Expected four completed replicas, got {len(completed)}')
    return completed


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();finalize(args.root)
