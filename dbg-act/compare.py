import torch
import os
from collections import defaultdict
from functools import lru_cache
from functools import partial
param_groups = defaultdict(list)
files = [x for x in os.listdir(".") if x.endswith(".pt")]
files = sorted(files, key=lambda x: os.path.getmtime(x))

for f in files:
    param_name = f.split("%")[0]
    param_groups[param_name].append(f)

@lru_cache
def load_tensor(f):
    return torch.load(f)

def load_two_tensors_from_files(files):
    if len(files) == 2:
        return load_tensor(files[0]), load_tensor(files[1])
    
    '''
    attn__key_states%4%0%1.pt
    attn__key_states%4%1%1.pt
    attn__key_states%4%2%1.pt
    attn__key_states%4%3%1.pt
    attn__key_states%1.pt
    '''
    # suffix is <tp>%<rank>%<dim_to_gather>
    # sort the above files and left will be %1.pt and the right is the concat version along the gather dim
    def comparator(x):
        x = x[:-3] # remove the .pt
        parts = x.split("%")
        return (int(parts[1]), int(parts[2]) if len(parts) > 2 else None, int(parts[3]) if len(parts) > 3 else None)
    files = sorted(files, key=comparator)
    left = load_tensor(files[0])
    right = [load_tensor(f) for f in files[1:]]
    dim_to_gather = comparator(files[-1])[2]  # just check the last
    right = torch.cat(right, dim=dim_to_gather)
    return left, right


N = None
def assert_close_slice(left, right, *indices):
    slc = []
    for i in indices:
        if isinstance(i, int):
            slc.append(i)
        elif isinstance(i, (tuple, list)):
            slc.append(slice(*i))
        elif isinstance(i, slice):
            slc.append(i)
        elif i == None:
            slc.append(slice(None))
        else:
            raise ValueError(f"Invalid index: {i}, type: {type(i)}")
    try:
        torch.testing.assert_close(left[tuple(slc)], right[tuple(slc)])
        print("ALL GOOD")
    except AssertionError as e:
        print(e)

def query_slice(left, right, *indices):
    slc = []
    for i in indices:
        if isinstance(i, int):
            slc.append(i)
        elif isinstance(i, (tuple, list)):
            slc.append(slice(*i))
        elif isinstance(i, slice):
            slc.append(i)
        elif i == None:
            slc.append(slice(None))
        else:
            raise ValueError(f"Invalid index: {i}, type: {type(i)}")
    print(left[tuple(slc)], right[tuple(slc)])

# can you sort and create a new param_Groups that's sorted by mtime? assuming each param group is:
# param_name: [file1, file2]
# sort is so that it is ascending by mtime min(mtime(file1), mtime(file2))
# Sort param_groups by the minimum mtime of files in each group (ascending)
param_groups = {k: sorted(v) for k, v in sorted(
    param_groups.items(),
    key=lambda x: min(os.path.getmtime(f) for f in x[1])
)}

for param_name, files in param_groups.items():
    print("="*50)
    print(f"Comparing {param_name}: {files}")
    print("="*50)
    
    # for each pair of files, compare the tensor values with torch.testing.allclose
    '''
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            left = load_tensor(files[i])
            right = load_tensor(files[j])
            '''
    if True:
            left, right = load_two_tensors_from_files(files)
            if isinstance(left, tuple):
                print(f"SHAPES: {tuple(left_i.shape for left_i in left)} vs {tuple(right_i.shape for right_i in right)}")
                for k, (left_i, right_i) in enumerate(zip(left, right)):
                    helper = partial(assert_close_slice, left_i, right_i)
                    try:
                        l2_norm = torch.norm(left_i.to(torch.float32) - right_i.to(torch.float32))
                        print(f"[{k}] L2 norm / numel: {l2_norm} / {left_i.numel()} = {l2_norm / left_i.numel()}")
                        torch.testing.assert_close(left_i, right_i)
                        print(f"ALL GOOD {k}")
                    except AssertionError as e:
                        print(e)
            else:
                print(f"SHAPES: {left.shape} vs {right.shape}")
                helper = partial(assert_close_slice, left, right)
                query = partial(query_slice, left, right)
                try:
                    # Compute L2 norm
                    l2_norm = torch.norm(left.to(torch.float32) - right.to(torch.float32))
                    print(f"L2 norm / numel: {l2_norm} / {left.numel()} = {l2_norm / left.numel()}")
                    torch.testing.assert_close(left, right)
                    print("ALL GOOD")
                except AssertionError as e:
                    if 'o_proj' in param_name:
                        #breakpoint()
                        import matplotlib.pyplot as plt
                        import numpy as np
                        from matplotlib.colors import SymLogNorm

                        # Compute the difference
                        diff = (left.to(torch.float32) - right.to(torch.float32)).cpu().numpy()
                        left_np = left.to(torch.float32).cpu().numpy()
                        right_np = right.to(torch.float32).cpu().numpy()
                        # diff, left_np, right_np shape: (2, 1560, 1024)

                        # --- Heatmaps as subplots ---
                        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
                        absmax = max(np.abs(diff).max(), 1e-8)
                        linthresh = absmax * 1e-3
                        norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-absmax, vmax=absmax, base=10)

                        for idx in range(diff.shape[0]):
                            ax = axes[idx]
                            im = ax.imshow(
                                diff[idx],
                                aspect='auto',
                                cmap='bwr',
                                norm=norm
                            )
                            vmin = np.min(diff[idx])
                            vmax = np.max(diff[idx])
                            ax.set_title(
                                f"Heatmap of {param_name} (left - right) [batch {idx}]\n"
                                f"(min={vmin:.2e}, max={vmax:.2e}, absmax={absmax:.2e})"
                            )
                            ax.set_xlabel("Feature dimension")
                            ax.set_ylabel("Sequence position")
                        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.04, label='Difference (symlog)')
                        plt.tight_layout()
                        out_path = f"heatmap_{param_name.replace('/', '_')}_batches.png"
                        plt.savefig(out_path)
                        print(f"Saved heatmap subplot to {out_path}")
                        plt.close(fig)

                        # --- Combined linearized error vs. data plots ---
                        # Flatten all batches
                        left_flat_0 = left_np[0].flatten()
                        right_flat_0 = right_np[0].flatten()
                        diff_flat_0 = diff[0].flatten()
                        left_flat_1 = left_np[1].flatten()
                        right_flat_1 = right_np[1].flatten()
                        diff_flat_1 = diff[1].flatten()
                        x0 = np.arange(left_flat_0.size)
                        x1 = np.arange(left_flat_1.size)

                        # Helper to plot a given slice
                        def plot_linearized_slice(slice_obj, tag):
                            fig, ax1 = plt.subplots(figsize=(16, 5))
                            ax1.plot(x0[slice_obj], left_flat_0[slice_obj], label='left[0]', alpha=0.7)
                            ax1.plot(x0[slice_obj], right_flat_0[slice_obj], label='right[0]', alpha=0.7)
                            ax1.plot(x1[slice_obj], left_flat_1[slice_obj], label='left[1]', alpha=0.7, linestyle='dashed')
                            ax1.plot(x1[slice_obj], right_flat_1[slice_obj], label='right[1]', alpha=0.7, linestyle='dashed')
                            ax1.set_ylabel('Value')
                            ax1.set_xlabel('Flattened index')
                            ax1.legend(loc='upper left', ncol=2)
                            ax1.set_title(f"Linearized Data and Error for {param_name} [{tag}]")

                            ax2 = ax1.twinx()
                            ax2.plot(x0[slice_obj], diff_flat_0[slice_obj], color='red', label='error[0] (left-right)', alpha=0.5)
                            ax2.plot(x1[slice_obj], diff_flat_1[slice_obj], color='orange', label='error[1] (left-right)', alpha=0.5)
                            ax2.set_ylabel('Error', color='red')
                            ax2.tick_params(axis='y', labelcolor='red')
                            ax2.legend(loc='upper right')

                            plt.tight_layout()
                            out_path = f"linearized_{param_name.replace('/', '_')}_{tag}.png"
                            plt.savefig(out_path)
                            print(f"Saved linearized error/data plot to {out_path}")
                            plt.close(fig)

                        # Plot for all, first 10000, first 1000
                        plot_linearized_slice(slice(None), "all")
                        plot_linearized_slice(slice(0, 10000), "first10000")
                        plot_linearized_slice(slice(0, 1000), "first1000")
                        plt.close()
                    print(e)