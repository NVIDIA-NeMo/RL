# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only tensor instrumentation for the first real Flint refit."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
from datetime import datetime, timezone


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(actual, expected):
    import torch
    a, b = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    result = {'actual_dtype': str(a.dtype), 'original_dtype': str(b.dtype),
              'actual_shape': list(a.shape), 'original_shape': list(b.shape),
              'actual_sha256': digest(a), 'original_sha256': digest(b)}
    if a.shape != b.shape:
        return {**result, 'value_equal': False, 'shape_match': False}
    a, b = a.reshape(-1), b.reshape(-1)
    if torch.equal(a, b) and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()):
        return {**result, 'shape_match': True, 'numel': a.numel(),
                'differing_elements': 0, 'nonfinite_elements': 0, 'value_equal': True,
                'max_abs_error': 0.0, 'mean_abs_error': 0.0,
                'actual_sample': a[:8].float().tolist(), 'original_sample': b[:8].float().tolist()}
    mismatch = nonfinite = 0
    max_abs = total_abs = 0.0
    for start in range(0, a.numel(), 1048576):
        x, y = a[start:start+1048576].double(), b[start:start+1048576].double()
        finite = torch.isfinite(x) & torch.isfinite(y)
        nonfinite += int((~finite).sum())
        mismatch += int((x != y).sum())
        delta = (x[finite]-y[finite]).abs()
        if delta.numel():
            max_abs = max(max_abs, float(delta.max()))
            total_abs += float(delta.sum())
    result.update(shape_match=True, numel=a.numel(), differing_elements=mismatch,
                  nonfinite_elements=nonfinite, value_equal=mismatch == 0 and nonfinite == 0,
                  max_abs_error=max_abs, mean_abs_error=total_abs/max(1,a.numel()),
                  actual_sample=a[:8].float().tolist(), original_sample=b[:8].float().tolist())
    return result


class Source:
    def __init__(self, root):
        from safetensors import safe_open
        self.root = Path(root)
        self.index = json.loads((self.root/'model.safetensors.index.json').read_text())['weight_map']
        self.handles = {f:safe_open(str(self.root/f),framework='pt',device='cpu') for f in set(self.index.values())}

    def get(self, name):
        return self.handles[self.index[name]].get_tensor(name)


class Dump:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.data, self.index, self.size, self.part = {}, {}, 0, 0

    def add(self, name, value):
        tensor = value.detach().cpu().contiguous().clone()
        self.data[name] = tensor
        self.size += tensor.numel()*tensor.element_size()
        if self.size >= 256*1024**2:
            self.flush()

    def flush(self):
        from safetensors.torch import save_file
        if not self.data:
            return
        name=f'weights-{self.part:04d}.safetensors'
        save_file(self.data,str(self.root/name))
        self.index.update({key:name for key in self.data})
        self.part += 1
        self.data, self.size = {}, 0

    def finish(self):
        self.flush()
        write(self.root/'model.safetensors.index.json',{'weight_map':self.index})


def expert_second_half(kind, load_up_first, flashinfer_trtllm):
    # _weight_loader_impl swaps w1/w3 for unquantized FlashInfer TRTLLM,
    # then _load_w13 applies the quant method's independent ordering flag.
    if flashinfer_trtllm:
        kind = {'gate_proj': 'up_proj', 'up_proj': 'gate_proj'}[kind]
    return (kind == 'gate_proj' and load_up_first) or (kind == 'up_proj' and not load_up_first)


def resident_view(model, source_name, source):
    """Independent TP1 slice mapping, before SGLang kernel-layout packing."""
    params = dict(model.named_parameters())
    name=source_name.replace('.mlp.gate.e_score_correction_bias','.mlp.gate.expert_bias')
    if name in params:
        return name, params[name]
    match=re.fullmatch(r'(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight',name)
    if match:
        prefix, expert, kind=match.groups()
        module=model.get_submodule(prefix)
        if getattr(module,'use_triton_kernels',False):
            raise RuntimeError('This audit supports the pinned FlashInfer layout only')
        target=prefix+('.w2_weight' if kind=='down_proj' else '.w13_weight')
        tensor=params[target][int(expert)]
        if kind!='down_proj':
            switch=bool(getattr(module.quant_method,'load_up_proj_weight_first',False))
            half=tensor.shape[0]//2
            second=expert_second_half(kind,switch,bool(getattr(module,'use_flashinfer_trtllm_moe',False)))
            tensor=tensor[half if second else 0:2*half if second else half]
        return target,tensor
    for piece in ('q_proj','k_proj','v_proj'):
        token='.'+piece+'.'
        if token in name:
            target=name.replace(token,'.qkv_proj.')
            sizes=[source.get(source_name.replace(piece,p)).shape[0] for p in ('q_proj','k_proj','v_proj')]
            slot=('q_proj','k_proj','v_proj').index(piece)
            start=sum(sizes[:slot])
            return target,params[target][start:start+sizes[slot]]
    for piece,slot in [('gate_proj',0),('up_proj',1)]:
        token='.'+piece+'.'
        if token in name:
            target=name.replace(token,'.gate_up_proj.')
            half=params[target].shape[0]//2
            return target,params[target][slot*half:(slot+1)*half]
    raise RuntimeError('Unmapped source tensor: '+source_name)


def canonical_audit(model, source, root, dump=False):
    records, mapped = [], set()
    output=Dump(root/'tensors') if dump else None
    for name in sorted(source.index):
        target,actual=resident_view(model,name,source)
        original=source.get(name)
        row={'name':name,'resident_parameter':target,**compare(actual,original)}
        records.append(row)
        mapped.add(target)
        if output:
            output.add(name,actual)
    extras=sorted(set(dict(model.named_parameters()))-mapped)
    if output:
        output.finish()
    report={'status':'pass' if all(x['value_equal'] for x in records) and not extras else 'mismatch',
            'comparison':'All canonical resident weight values versus original four-layer input, before kernel packing',
            'tensor_count':len(records),'numel':sum(x.get('numel',0) for x in records),
            'unmapped_resident_parameters':extras,'records':records}
    write(root/'comparison.json',report)
    return report


def install():
    import flint_sglang.refit as refit
    original_install=refit.install_runner_refit_hooks

    def install_hooks(model_type, runner_type=None):
        original_install(model_type,runner_type)
        if runner_type is None:
            from sglang.srt.model_executor.model_runner import ModelRunner
            runner_type=ModelRunner
        if getattr(runner_type.begin_weight_update,'_step0_audit',False):
            return
        original_begin,original_end=runner_type.begin_weight_update,runner_type.end_weight_update
        original_load,original_forward=model_type.load_weights,model_type.forward

        def root_for(model):
            if not hasattr(model,'_step0_root'):
                model._step0_root=Path(os.environ['FLINT_STEP0_OUTPUT'])/f'{socket.gethostname()}-{os.getpid()}'
                model._step0_source=Source(os.environ['FLINT_STEP0_SOURCE'])
                model._step0_refits=0
                model._step0_forward_count=0
            return model._step0_root

        def load(model,weights,*args,**kwargs):
            root=root_for(model)
            first_refit=model._refit_coverage.active and model._step0_refits==1
            records=[]
            def inspected():
                for name,tensor in weights:
                    if first_refit:
                        source_name=name.replace('.mlp.gate.expert_bias','.mlp.gate.e_score_correction_bias')
                        if source_name not in model._step0_source.index and name in model._step0_source.index:
                            source_name=name
                        records.append({'name':source_name,**compare(tensor,model._step0_source.get(source_name))})
                    yield name,tensor
            result=original_load(model,inspected(),*args,**kwargs)
            if first_refit:
                with (root/'incoming-refit.jsonl').open('a') as f:
                    for row in records: f.write(json.dumps(row,allow_nan=False)+'\n')
            elif not model._refit_coverage.active:
                canonical_audit(model,model._step0_source,root/'initial-logical')
            return result

        def begin(runner):
            if not isinstance(runner.model,model_type):
                return original_begin(runner)
            model=runner.model;root=root_for(model)
            model._step0_refits+=1
            if model._step0_refits==1:
                import torch
                torch.cuda.synchronize()
                root.mkdir(parents=True,exist_ok=True)
                model._step0_initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
                model._step0_forward_at_begin=model._step0_forward_count
                write(root/'started.json',{'phase':'first_refit_begin','timestamp_utc':datetime.now(timezone.utc).isoformat(),
                    'forwards_before_initial_refit':model._step0_forward_count,'reference':'Initial SGLang checkpoint load, after kernel packing'})
            return original_begin(runner)

        def end(runner,run_post_load):
            if not isinstance(runner.model,model_type) or runner.model._step0_refits!=1:
                return original_end(runner,run_post_load)
            model=runner.model;root=root_for(model)
            logical=canonical_audit(model,model._step0_source,root/'step0-logical',dump=True)
            result=original_end(runner,run_post_load)
            import torch
            torch.cuda.synchronize()
            output=Dump(root/'step0-resident')
            packed=[]
            for name,param in model.named_parameters():
                output.add(name,param)
                packed.append({'name':name,**compare(param,model._step0_initial[name])})
            output.finish()
            incoming=[json.loads(line) for line in (root/'incoming-refit.jsonl').read_text().splitlines()]
            incoming_names=[r['name'] for r in incoming]
            initial=json.loads((root/'initial-logical/comparison.json').read_text())
            report={'phase':'step0_after_refit_postprocessing_before_first_rollout','timestamp_utc':datetime.now(timezone.utc).isoformat(),
                'first_refit_index':1,'forwards_during_refit':model._step0_forward_count-model._step0_forward_at_begin,
                'initial_source_comparison':initial['status'],'logical_refit_comparison':logical['status'],
                'incoming_count':len(incoming),'incoming_unique_count':len(set(incoming_names)),
                'incoming_source_coverage':set(incoming_names)==set(model._step0_source.index),
                'incoming_values_equal':all(r['value_equal'] for r in incoming),
                'postprocessed_resident_values_equal_initial':all(r['value_equal'] for r in packed),
                'resident_parameter_count':len(packed),'resident_records':packed,
                'scope':'Complete incoming HF tensors and canonical resident slices compared directly to input checkpoint; postprocessed resident tensors compared to validated initial serving layout. No logits or optimizer equivalence claim.'}
            passed=(initial['status']=='pass' and logical['status']=='pass' and report['incoming_source_coverage'] and
                    len(incoming_names)==len(set(incoming_names)) and report['incoming_values_equal'] and
                    report['postprocessed_resident_values_equal_initial'] and report['forwards_during_refit']==0)
            report['status']='pass' if passed else 'mismatch'
            write(root/'step0-comparison.json',report)
            model._step0_initial=None
            model._step0_complete=True
            print('FLINT_STEP0_TENSOR_AUDIT '+str(root)+' '+report['status'],flush=True)
            return result

        def forward(model,*args,**kwargs):
            root=root_for(model)
            if model._step0_refits and not getattr(model,'_step0_complete',False):
                raise RuntimeError('Forward attempted before step0 audit completed')
            if model._step0_refits and not (root/'first-forward-after-audit.json').exists():
                write(root/'first-forward-after-audit.json',{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                    'audit_exists':(root/'step0-comparison.json').is_file(),'forward_index':model._step0_forward_count})
            model._step0_forward_count+=1
            return original_forward(model,*args,**kwargs)

        begin._step0_audit=True
        model_type.load_weights=load
        model_type.forward=forward
        runner_type.begin_weight_update=begin
        runner_type.end_weight_update=end

    refit.install_runner_refit_hooks=install_hooks
